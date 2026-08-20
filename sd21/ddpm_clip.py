"""OpenCLIP-conditioned latent diffusion — the SD2.1 arm.

Fork of ldm/models/diffusion/ddpm_clip.py; nothing in ldm/ is modified.
Two deltas vs the SD1.5 arm:

- Text tower is OpenCLIP ViT-H/14 via its HF-format mirror
  (laion/CLIP-ViT-H-14-laion2B-s32B-b79K, hidden size 1024) — the same
  weights SD2.1 froze; loads through transformers, no open_clip dependency.
- SD2.x conditions on the PENULTIMATE hidden layer (+ final layer norm),
  not last_hidden_state; `penultimate=True` reproduces that.

Original SD1.5-arm notes that still apply:

- get_input / shared_step / training_step / validation_step / CSV logging /
  _compute_validation_metrics / sample_log are inherited unchanged, so the
  training loop is bit-identical to the LLM and CT-CLIP arms'.
- eval_all.py's isinstance(model, LatentDiffusionCond_LLM) dispatch works
  without modification; the inherited code paths call
  self.text_encoder(tokens, contexts) where contexts is None here
  (unet_config.context_length: 0) and CLIPTextEncoder ignores it.
- Because context_dim is 768 (native SD width), init_from_ckpt loads the
  pretrained attn2.to_k/to_v weights from the unconditional checkpoint intact.
- max_length is 77 (CLIP's position-embedding limit), not the 128 the BERT
  arms use; dataset._encode_prompt reads it through the retriever protocol.
"""

import torch
import torch.nn as nn

from ldm.models.diffusion.ddpm import LatentDiffusion, LatentDiffusionCond_LLM
from ldm.modules.encoders.ctclip_encoder import TokenizerShim


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP ViT-L/14 text tower — the stock SD1.5 conditioning encoder.

    Forward maps dataset-tokenized prompts [B, L] to the full hidden-state
    sequence [B, L, 768] for UNet cross-attention, mirroring
    FrozenCLIPEmbedder: last_hidden_state, no attention mask (CLIP's causal
    attention keeps the dataset's 0-padding from touching earlier positions).

    The embedding table is resized by +1 because dataset._encode_prompt
    appends token id == vocab_size as a truncation marker (same trick as the
    MedGemma and CT-CLIP arms).
    """

    def __init__(self, version="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 penultimate=True):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel

        self.penultimate = penultimate
        self.hf_tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer.resize_token_embeddings(self.hf_tokenizer.vocab_size + 1)

        for param in self.transformer.parameters():
            param.requires_grad_(False)
        self.transformer.eval()

    def train(self, mode=True):
        # Permanently eval: a frozen conditioning tower must not flip to
        # train() when Lightning switches modes.
        return super().train(False)

    @torch.no_grad()
    def forward(self, text, context=None, **kwargs):
        # `context` accepted for signature parity with TextContextEncoder
        # (soft prompts); this arm has none, so it is ignored.
        out = self.transformer(input_ids=text,
                               output_hidden_states=self.penultimate)
        if self.penultimate:
            # hidden_states[-2] is the second-to-last block's output; SD2.x
            # applies the final LayerNorm on top of it (open_clip "penultimate").
            h = self.transformer.text_model.final_layer_norm(out.hidden_states[-2])
            return h.float()
        return out.last_hidden_state.float()


class LatentDiffusionCond_CLIP(LatentDiffusionCond_LLM):

    def __init__(self, *args,
                 clip_version="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 penultimate=True,
                 **kwargs):
        # Deliberately skip LatentDiffusionCond_LLM.__init__ (it loads MedGemma);
        # run the grandparent init directly, then attach the CLIP tower.
        LatentDiffusion.__init__(self, *args, **kwargs)

        self.context_length = self.model.diffusion_model.context_length
        self.n_prompts = self.model.diffusion_model.n_prompts
        assert self.context_length == 0, (
            "CLIP arm has no soft prompts; set unet_config.params.context_length: 0"
        )

        self.text_encoder = CLIPTextEncoder(clip_version, penultimate=penultimate)
        self.tokenizer = TokenizerShim(self.text_encoder.hf_tokenizer)
        # Retriever protocol: dataset._encode_prompt pads prompts to
        # max_length - context_length token slots. 77 is CLIP's position limit.
        self.max_length = 77

        # Per-epoch loss tracking for CSV logging (inherited hooks)
        self._train_losses = []
        self._val_losses = []
        self._csv_initialized = False

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        # Same shape-tolerant loader as the parent, with mmap=True so the 11GB
        # checkpoint (which includes optimizer state) never fully materializes.
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

        sd_dict = sd['state_dict']
        model_dict = self.model.state_dict()

        for name, weight in sd_dict.items():
            for name_model, weigh_model in model_dict.items():
                if name.endswith(name_model):
                    if model_dict[name_model].shape != weight.shape:
                        ignore_keys.append(name)
                        names_no_dot = name.replace(".", "")
                        names_no_dot = 'model_ema.' + names_no_dot[5:]
                        ignore_keys.append(names_no_dot)

        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)

        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
