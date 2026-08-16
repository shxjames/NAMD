"""Verify the NAMD stage-1 -> stage-2 checkpoint transfer is exact. Runs on CPU
in seconds; no GPU, no MedGemma download, no training.

Two independent checks:

  A. Config equivalence. The two configs are diffed directly, so any divergence
     in unet_config (context_dim, context_length, n_prompts, txt_embed_dim,
     model_channels, ...) or first_stage ddconfig is caught -- these are exactly
     the things that would make stage 2 silently drop weights again.

  B. Loader behaviour. Both models are instantiated at reduced width and stage 1
     is loaded into stage 2 through the strict loader, proving the 4 -> 8 input
     conv expansion works and that everything else transfers bit-identically.
     Width is reduced only so this runs on a login node; it does not affect key
     layout, which check A covers at full size.

MedGemma is stubbed: it is attached after init_from_ckpt in both stages and is
excluded from checkpoints, so it plays no part in the transfer.

    python scripts/check_stage_transfer.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ldm.models.diffusion.ddpm as ddpm_mod
from ldm.util import instantiate_from_config

STAGE1 = "configs/ldm/v1.5-unconditional-unet-medgemma.yaml"
STAGE2 = "configs/ldm/v1.5-conditional-unet-llm-strict.yaml"
CONV_IN = "model.diffusion_model.input_blocks.0.0.weight"
NARROW = 32  # model_channels / VAE ch for check B


STUB_VOCAB = 16


class _StubTransformer(nn.Module):
    """Identity stand-in for MedGemma: returns its input embeddings unchanged, so
    the pooled output names the token whose embedding sits at the pooled index."""

    def forward(self, inputs_embeds=None, **kw):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def stub_text_encoder(txt_embed_dim, llm_model=None, llm_load_in_4bit=False):
    """Real TextContextEncoder, stubbed LLM. Mirrors build_medgemma_text_encoder:
    token_embedding is replaced by one of width txt_embed_dim (MedGemma's
    embed_tokens in the real path) and the transformer is frozen."""
    from ldm.text_encoder import TextContextEncoder
    enc = TextContextEncoder(embed_dim=txt_embed_dim)
    enc.llm = True
    enc.llm_type = 'medgemma'
    enc.llm_compute_dtype = torch.float32
    emb = nn.Embedding(STUB_VOCAB, txt_embed_dim)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(STUB_VOCAB, dtype=torch.float32)
                         .unsqueeze(1).expand(STUB_VOCAB, txt_embed_dim))
    enc.token_embedding = emb
    enc.transformer = _StubTransformer()
    for p in enc.parameters():
        p.requires_grad_(False)
    tok = SimpleNamespace(bos_token_id=2, eos_token_id=1, pad_token_id=0)
    return enc, tok, 512


ddpm_mod.build_medgemma_text_encoder = stub_text_encoder


def check_configs():
    print("A. config equivalence")
    c1, c2 = OmegaConf.load(STAGE1), OmegaConf.load(STAGE2)
    p1, p2 = c1.model.params, c2.model.params
    u1 = OmegaConf.to_container(p1.unet_config.params, resolve=True)
    u2 = OmegaConf.to_container(p2.unet_config.params, resolve=True)

    # use_checkpoint is gradient checkpointing; it changes no parameter shape.
    ignore = {"in_channels", "use_checkpoint"}
    diff = {k: (u1.get(k), u2.get(k))
            for k in set(u1) | set(u2)
            if k not in ignore and u1.get(k) != u2.get(k)}
    assert not diff, f"unet_config diverges (stage1, stage2): {diff}"
    assert (u1["in_channels"], u2["in_channels"]) == (4, 8), \
        f"expected in_channels 4 -> 8, got {u1['in_channels']} -> {u2['in_channels']}"
    print(f"   unet_config    : identical apart from in_channels 4 -> 8")
    print(f"   context_dim={u1['context_dim']}  context_length={u1['context_length']}  "
          f"n_prompts={u1['n_prompts']}  txt_embed_dim={u1['txt_embed_dim']}")

    d1 = OmegaConf.to_container(p1.first_stage_config.params.ddconfig, resolve=True)
    d2 = OmegaConf.to_container(p2.first_stage_config.params.ddconfig, resolve=True)
    assert d1 == d2, f"first_stage ddconfig diverges: {d1} vs {d2}"
    assert p1.first_stage_config.params.embed_dim == p2.first_stage_config.params.embed_dim
    print("   first_stage    : ddconfig identical")

    # scale_factor is a config value, not a state_dict entry, so no load can
    # catch a mismatch -- it has to be checked here.
    assert p1.scale_factor == p2.scale_factor, \
        f"scale_factor {p1.scale_factor} != {p2.scale_factor}"
    print(f"   scale_factor   : {p1.scale_factor} on both")
    assert p1.conditioning_mode == "llm" and p2.strict_load
    print("   modes          : stage1 conditioning_mode=llm, stage2 strict_load=True")
    for p, name in ((p1, "stage1"), (p2, "stage2")):
        assert "cond_stage_config" not in p, f"{name} still declares a CLIP cond stage"
    print("   cond_stage     : absent in both (no CLIP)")


def check_empty_prompt(m1):
    """The stage-1 conditioning really is soft prompts terminated by <EOS>.

    With the identity transformer, the pooled vector is just the embedding of
    whichever token sits at the pooled index, and token id i embeds to the
    constant vector i -- so the value read back names the pooled token.
    """
    ids = m1._empty_prompt_ids()
    assert ids.tolist() == [[2, 1]], f"expected [[BOS, EOS]], got {ids.tolist()}"

    emb = m1.get_null_conditioning(4)
    assert emb.shape == (4, m1.n_prompts, m1.txt_embed_dim), \
        f"expected (4, {m1.n_prompts}, {m1.txt_embed_dim}), got {tuple(emb.shape)}"
    pooled = {int(round(v)) for v in emb[:, :, 0].flatten().tolist()}
    assert pooled == {1}, f"pooled token id(s) {pooled}, expected the EOS id 1"
    # Sequence seen by the LLM: [BOS] + n_prompts x context_length + [EOS].
    print(f"   empty prompt   : [BOS] + {m1.n_prompts}x{m1.context_length} soft "
          f"prompts + [EOS], pooled at <eos>, context {tuple(emb.shape)}")


def narrow(cfg):
    cfg.model.params.unet_config.params.model_channels = NARROW
    cfg.model.params.first_stage_config.params.ddconfig.ch = NARROW
    return cfg


def check_loader():
    print("\nB. loader behaviour (reduced width)")
    c1 = narrow(OmegaConf.load(STAGE1))
    del c1.model.params.ckpt_path                      # skip the SD1.5 read
    del c1.model.params.first_stage_config.params.ckpt_path
    m1 = instantiate_from_config(c1.model)

    unet1 = m1.model.diffusion_model
    assert unet1.contexts.requires_grad, "train_contexts: True should leave contexts trainable"
    assert m1.conditioning_mode == "llm"
    assert m1.cond_stage_model is not None and not list(m1.cond_stage_model.parameters()), \
        "cond stage should be a parameter-free Identity, not CLIP"
    print(f"   contexts       : {tuple(unet1.contexts.shape)} trainable")

    m1.learning_rate = c1.model.base_learning_rate      # run.py sets this, not the config
    opt = m1.configure_optimizers()
    if isinstance(opt, (list, tuple)):                  # ([opt], [sched]) when use_scheduler
        opt = opt[0][0]
    trainable = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(unet1.contexts) in trainable, "contexts missing from the optimizer"
    print("   optimizer      : contexts included")

    check_empty_prompt(m1)

    sd1 = {k: v.clone() for k, v in m1.state_dict().items()}
    assert not [k for k in sd1 if k.startswith("cond_stage_model.")], \
        "no CLIP weights should reach the checkpoint"
    assert [k for k in sd1 if k.startswith("text_encoder.")], \
        "the stub encoder should be in state_dict, so the strip hook is exercised"
    ckpt = {"state_dict": dict(sd1)}
    m1.on_save_checkpoint(ckpt)                        # exercise the strip hook
    assert not [k for k in ckpt["state_dict"] if k.startswith("text_encoder.")]
    print(f"   ckpt strip     : {len(sd1) - len(ckpt['state_dict'])} text_encoder "
          f"tensors dropped from the checkpoint")
    del m1

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "stage1.ckpt")
        torch.save(ckpt, path)
        c2 = narrow(OmegaConf.load(STAGE2))
        OmegaConf.update(c2, "model.params.ckpt_path", path)
        m2 = instantiate_from_config(c2.model)         # raises if load is not exact

    sd2 = m2.state_dict()
    w1, w2 = sd1[CONV_IN], sd2[CONV_IN]
    n = w1.shape[1]
    assert torch.equal(w2[:, :n], w1), "pretrained input-conv channels were not copied"
    assert torch.count_nonzero(w2[:, n:]) == 0, "concat channels were not zero-initialised"
    print(f"   input conv     : {tuple(w1.shape)} -> {tuple(w2.shape)}, "
          f"first {n} copied, last {w2.shape[1] - n} zero")

    # text_encoder is excluded from checkpoints and rebuilt per run, so the two
    # stubs are independently initialised and are not expected to agree.
    shared = [k for k in sd1
              if k in sd2 and k != CONV_IN and not k.startswith("text_encoder.")]
    differing = [k for k in shared if not torch.equal(sd1[k], sd2[k])]
    assert not differing, f"{len(differing)} tensors differ after load: {differing[:10]}"
    kv = [k for k in shared if "attn2.to_k" in k or "attn2.to_v" in k]
    assert len(kv) == 32, f"expected 32 cross-attention K/V matrices, found {len(kv)}"
    assert "model.diffusion_model.contexts" in shared
    print(f"   transferred    : {len(shared)} tensors bit-identical, "
          f"including all {len(kv)} attn2.to_k/to_v and contexts")


if __name__ == "__main__":
    check_configs()
    check_loader()
    print("\nstage-1 -> stage-2 transfer is exact.")
