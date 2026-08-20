"""Report len(train_dataloader()) for a config, so the LR schedule can be sized.

LambdaWarmUpCosineScheduler2 takes ABSOLUTE step counts. cycle_lengths must
exceed the run's total steps -- find_in_interval returns None past the last
cycle and the caller then indexes with it -- but stay close to it, or the cosine
never advances and the LR sits flat at f_max with no decay. Both failure modes
have already bitten this repo, so derive the number instead of guessing it.

    python -m scripts.sd21_namd.steps_per_epoch \
        --config configs/ldm/v2.1-unconditional-unet-namd.yaml

Builds only the datamodule, never the model, so it runs on CPU.
"""
import argparse

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


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
    dm.setup()
    steps = len(dm.train_dataloader())

    epochs = int(config.trainer.max_epochs)
    total = steps * epochs

    print(f"batch_size      = {config.data.params.batch_size}")
    print(f"max_epochs      = {epochs}")
    print(f"STEPS_PER_EPOCH = {steps}")
    print(f"TOTAL_STEPS     = {total}")
    print(f"CYCLE_LENGTHS   = {total + steps}   # total + 1 epoch headroom")


if __name__ == "__main__":
    main()
