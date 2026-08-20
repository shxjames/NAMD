"""SD2.1 unconditional-stage text encoder.

OpenCLIP ViT-H/14 text tower via its HF-format mirror
(laion/CLIP-ViT-H-14-laion2B-s32B-b79K) — the same weights SD2.1 froze —
loaded through transformers, so no open_clip dependency. SD2.x conditions on
the penultimate hidden layer with the final LayerNorm applied on top
(open_clip's layer="penultimate"); hidden_states[-2] + final_layer_norm is
the HF equivalent.
"""

from ldm.modules.encoders.modules import FrozenCLIPEmbedder


class FrozenOpenCLIPHFEmbedder(FrozenCLIPEmbedder):

    def __init__(self, version="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 device="cuda", max_length=77, penultimate=True):
        super().__init__(version=version, device=device, max_length=max_length)
        self.penultimate = penultimate

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        return_length=True, return_overflowing_tokens=False,
                                        padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.penultimate)
        if self.penultimate:
            return self.transformer.text_model.final_layer_norm(outputs.hidden_states[-2])
        return outputs.last_hidden_state
