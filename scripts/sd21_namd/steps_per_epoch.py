"""Report len(train_dataloader()) for a config, so the LR schedule can be sized.

LambdaWarmUpCosineScheduler2 takes ABSOLUTE step counts. cycle_lengths must
exceed the run's total steps -- find_in_interval returns None past the last
cycle and the caller then indexes with it -- but stay close to it, or the cosine
never advances and the LR sits flat at f_max with no decay. Both failure modes
have already bitten this repo, so derive the number instead of guessing it.

    python -m scripts.sd21_namd.steps_per_epoch \
        --config configs/ldm/v2.1-unconditional-unet-namd.yaml

Builds only the datamodule, never the diffusion model, so it runs on CPU.

The conditional dataset builds a prompt per subject and needs a retriever --
run.py supplies the model itself via dm.set_retriever(model). Constructing the
model here would mean loading MedGemma just to count batches, so this stands up
a stub exposing the same three things Demographic_Features._encode_prompt reads:
.max_length, .context_length and a .tokenizer with .encode()/.eos_token_id. The
tokenizer is the identical AutoTokenizer training builds
(ddpm.py:2024), so the sample count matches the real run rather than
approximating it.

Without a retriever every subject raises inside Demographic_Features, and the
dataset's per-subject `except ... continue` swallows all of them -- leaving zero
images and a bare "need at least one array to stack" from np.stack. That failure
is silent about its cause, so the emptiness check below reports it directly.
"""
import argparse

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class _PromptRetriever:
    """Minimal stand-in for the model when all the dataset wants is a tokenizer."""

    def __init__(self, llm_model, context_length, max_length=128):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.context_length = context_length
        self.max_length = max_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--set', type=str, nargs='*', default=[],
                        help='OmegaConf dotlist overrides, e.g. data.params.batch_size=8')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))

    dm = instantiate_from_config(config.data)

    if config.data.params.get('construct_prompt', False):
        mp = config.model.params
        retriever = _PromptRetriever(
            llm_model=mp.llm_model,
            context_length=mp.unet_config.params.context_length,
        )
        dm.set_retriever(retriever)
        print(f"retriever stub: {mp.llm_model}, "
              f"max_length={retriever.max_length}, "
              f"context_length={retriever.context_length}")

    dm.setup()

    n_train = len(dm.train_dataset)
    if n_train == 0:
        raise SystemExit(
            "train split built 0 samples. Every subject was dropped by the "
            "dataset's per-subject exception handler -- usually a missing or "
            "mismatched retriever when construct_prompt is set."
        )

    steps = len(dm.train_dataloader())
    epochs = int(config.trainer.max_epochs)
    total = steps * epochs

    print(f"train samples   = {n_train}")
    print(f"batch_size      = {config.data.params.batch_size}")
    print(f"max_epochs      = {epochs}")
    print(f"STEPS_PER_EPOCH = {steps}")
    print(f"TOTAL_STEPS     = {total}")
    print(f"CYCLE_LENGTHS   = {total + steps}   # total + 1 epoch headroom")


if __name__ == "__main__":
    main()
