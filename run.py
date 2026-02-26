import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import torch
import matplotlib.pyplot as plt
import os
import wandb

import lpips
import glob

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="number of epochs",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="checkpoint to load",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="whether to train dat shit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="test",
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample",
    )

    parser.add_argument(
        "--sample_from_train",
        action="store_true",
        help="sample from train set",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="batch size",
    )

    parser.add_argument(
        "--config", 
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=3,
        help="number of samples to generate for autoencoder reconstruction",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="precision",
    )

    parser.add_argument(
        "--monitor_loss",
        type=str,
        default="val/loss",
        help="monitor loss",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=-1,
        help="early stopping patience (number of epochs with no improvement). Set to -1 to disable early stopping.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="wandb project name",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="wandb entity (username or team name)",
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="wandb run name (defaults to project name if not specified)",
    )

    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=[],
        help="wandb tags for the run",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable wandb logging",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="run wandb in offline mode",
    )

    return parser

def sample_img(tensors: list[torch.Tensor], legend=['Condition', 'Reconstruction'], save_path='samples', greyscale=False, labels=None):
    os.makedirs(save_path, exist_ok=True)

    assert len(tensors) == len(legend)

    def unnormalize(tensor):
        tensor = (tensor + 1.0) / 2.0 * 255.0
        return tensor.to(torch.uint8)
    
    def make_greyscale(tensor):
        return torch.mean(tensor, dim=1, keepdim=True)
    
    for i, tensor in enumerate(tensors):
        if tensor.shape[1] == 3 and greyscale:
            tensor = make_greyscale(tensor)
        tensor = unnormalize(tensor)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensors[i] = tensor  # Update the list in place

    num_samples = tensors[0].shape[0]

    for i in range(num_samples):
        plt.figure(figsize=(10, 5))

        for j, tensor in enumerate(tensors):
            tensor = tensor[i]
            tensor = tensor.cpu().numpy().transpose(1, 2, 0)

            plt.subplot(1, len(tensors), j + 1)
            plt.imshow(tensor, cmap='gray')
            plt.axis('off')
            plt.title(legend[j])
        
        if labels is not None:
            plt.title(f"Label: {labels[i].item()}")
        plt.savefig(f'{save_path}/sample_{i}.png')
        plt.close()

def test_lpips(test_loader, model, device='cuda'):
    model.eval()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)


    lpips_loss_values = 0
    count = 0
    for batch in test_loader:
        cond_img, label = model.get_input(batch)

        with torch.no_grad():
            dec, posterior, z = model(cond_img)

            lpips_loss_value = lpips_loss(cond_img, dec)
            count += lpips_loss_value.shape[0]
            lpips_loss_values += lpips_loss_value.cpu().sum().item()
            

    return lpips_loss_values / count

def test_lpips_ldm(test_loader, model, device='cuda'):
    model.eval()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)


    lpips_loss_values = 0
    count = 0

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=True)

    for batch in test_loader:
        (cond_img, ehr_c), (gt_img, ehr_gt), label = batch

        cond_img = cond_img.to(device).to(model.dtype)
        ehr_c = ehr_c.to(device).to(model.dtype)
        gt_img = gt_img.to(device).to(model.dtype)

        with torch.no_grad():
            ddim_start = torch.randn(cond_img.shape[0], 16, 8, 8).to(device).to(model.dtype)

            samples_ddim = sampler.decode(
                ddim_start,
                t_start=49,
                sc = None,
                lung=True,
                ehr_c=ehr_c,
                x_ref=cond_img,
            )
            samples_ddim = model.decode_first_stage(samples_ddim)

            lpips_loss_value = lpips_loss(gt_img, samples_ddim)

            count += lpips_loss_value.shape[0]
            lpips_loss_values += lpips_loss_value.cpu().sum().item()
            

    return lpips_loss_values / count

def get_best_checkpoint(project_name, checkpoint_dir="checkpoints"):
    """
    Get the best checkpoint path for a given project
    """
    import os
    import glob
    
    project_path = os.path.join(checkpoint_dir, project_name)
    if not os.path.exists(project_path):
        return None
    
    # Find all checkpoint files
    checkpoint_folders = glob.glob(os.path.join(project_path, "epoch=*"))
    
    if not checkpoint_folders:
        return None
    
    # Extract scores from filenames (assuming format: epoch-val_loss.ckpt)
    checkpoint_scores = []
    for folder in checkpoint_folders:
        checkpoint_files = glob.glob(os.path.join(folder, "*.ckpt"))

        if not checkpoint_files:
            continue

        for ckpt_file in checkpoint_files:
            filename = os.path.basename(ckpt_file)
            try:
                # Extract loss value from filename like "epoch=09-val/rec_loss=4.99.ckpt"
                if "rec_loss=" in filename:
                    score_str = filename.split("rec_loss=")[1].split(".ckpt")[0]
                    score = float(score_str)
                    checkpoint_scores.append((score, ckpt_file))
                elif "loss=" in filename:
                    score_str = filename.split("loss=")[1].split(".ckpt")[0]
                    score = float(score_str)
                    checkpoint_scores.append((score, ckpt_file))
            except:
                continue

    if not checkpoint_scores:
        return None
    
    # Return the checkpoint with the lowest loss (best model)
    best_checkpoint = min(checkpoint_scores, key=lambda x: x[0])[1]
    return best_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = get_parser()
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)

    # Data module no longer needs llm_config - tokenization is done on-the-fly in the model
    dm = instantiate_from_config(config.data)
    dm.set_retriever(model)

    seed_everything(args.seed)

    # Handle resume from checkpoint BEFORE creating logger
    # so we can extract the wandb run ID for continuation
    resume_from_checkpoint = None
    resume_wandb_id = None
    
    if args.resume:
        # Determine checkpoint path
        if os.path.isfile(args.resume):
            # Direct path to checkpoint file
            resume_from_checkpoint = args.resume
            print(f"Resuming training from checkpoint: {args.resume}")
        elif os.path.isdir(args.resume):
            # Directory containing checkpoints - find last.ckpt first, then latest
            last_ckpt = os.path.join(args.resume, "last.ckpt")
            if os.path.exists(last_ckpt):
                resume_from_checkpoint = last_ckpt
                print(f"Resuming training from last checkpoint: {last_ckpt}")
            else:
                ckpt_files = glob.glob(os.path.join(args.resume, "**/*.ckpt"), recursive=True)
                if ckpt_files:
                    # Sort by modification time to get the latest
                    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                    resume_from_checkpoint = latest_ckpt
                    print(f"Resuming training from latest checkpoint: {latest_ckpt}")
                else:
                    print(f"No checkpoint files found in directory: {args.resume}")
        else:
            # Try to find checkpoint by project name
            checkpoint_dir = f"checkpoints/{args.resume}"
            if os.path.exists(checkpoint_dir):
                last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    resume_from_checkpoint = last_ckpt
                    print(f"Resuming training from last checkpoint: {last_ckpt}")
                else:
                    project_ckpt = get_best_checkpoint(args.resume)
                    if project_ckpt:
                        resume_from_checkpoint = project_ckpt
                        print(f"Resuming training from best checkpoint: {project_ckpt}")
            else:
                print(f"Could not find checkpoint for project: {args.resume}")
        
        # Try to extract wandb run ID from checkpoint directory for continuation
        if resume_from_checkpoint:
            ckpt_dir = os.path.dirname(resume_from_checkpoint)
            # Look for wandb run ID in the logs directory
            log_dir = os.path.join(args.logdir, args.project)
            if os.path.exists(log_dir):
                # Search for wandb run directories
                wandb_dirs = glob.glob(os.path.join(log_dir, "wandb", "run-*"))
                if wandb_dirs:
                    # Get the most recent wandb run
                    latest_wandb_dir = max(wandb_dirs, key=os.path.getmtime)
                    # Extract run ID from directory name (format: run-YYYYMMDD_HHMMSS-RUNID)
                    run_id_file = os.path.join(latest_wandb_dir, "run-*.wandb")
                    run_files = glob.glob(run_id_file)
                    if run_files:
                        # Extract run ID from filename
                        run_file = os.path.basename(run_files[0])
                        resume_wandb_id = run_file.split('-')[1].split('.')[0]
                        print(f"Found previous wandb run ID: {resume_wandb_id}")
            
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING")
            print(f"{'='*60}")
            print(f"Checkpoint: {resume_from_checkpoint}")
            print(f"Project: {args.project}")
            if resume_wandb_id:
                print(f"Wandb run ID: {resume_wandb_id} (will continue logging)")
            print(f"{'='*60}\n")

    # Set up logger (wandb or None)
    if args.wandb:
        wandb_run_name = args.wandb_name if args.wandb_name else args.project
        
        # If resuming, use the same run ID to continue logging
        logger_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": wandb_run_name,
            "tags": args.wandb_tags,
            "offline": args.offline,
            "save_dir": args.logdir,
        }
        
        # Add resume parameter if we found a previous run ID
        if resume_wandb_id:
            logger_kwargs["id"] = resume_wandb_id
            logger_kwargs["resume"] = "must"  # Must resume this run
            print(f"Continuing wandb run: {resume_wandb_id}")
        
        logger = WandbLogger(**logger_kwargs)
        
        # Log configuration to wandb and persist full config copies
        # Note: In DDP, logger.experiment may not be fully initialized until after trainer starts
        # We defer config logging to after training starts, or guard with hasattr check
        try:
            experiment = logger.experiment
            # Check if experiment is actually a wandb.Run (not a function or dummy)
            if hasattr(experiment, 'config') and hasattr(experiment.config, 'update'):
            # Log the full resolved config to the run
                full_config = OmegaConf.to_container(config, resolve=True)
                experiment.config.update(full_config, allow_val_change=True)
            # Save a copy into the wandb run directory
                run_dir = experiment.dir if hasattr(experiment, "dir") else args.logdir
                os.makedirs(run_dir, exist_ok=True)
                config_path = os.path.join(run_dir, "config.yaml")
                OmegaConf.save(config, config_path)
                
                # Upload config file to online wandb (if not offline)
                if not args.offline:
                    wandb.save(config_path)
        except Exception as e:
            print(f"Warning: failed to save config to wandb run dir: {e}")
    else:
        logger = None
        print("Wandb logging disabled (use --wandb to enable)")

    # Always save a copy alongside checkpoints and logs
    try:
        ckpt_dir = os.path.join("checkpoints", args.project)
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(ckpt_dir, "config.yaml"))

        log_proj_dir = os.path.join(args.logdir, args.project)
        os.makedirs(log_proj_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(log_proj_dir, "config.yaml"))
    except Exception as e:
        print(f"Warning: failed to persist config copies: {e}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.project}',
        filename=f'{{epoch:02d}}-{{{args.monitor_loss}:.3f}}',
        save_top_k=5,
        monitor=args.monitor_loss,
        mode='max' if 'auroc' in args.monitor_loss else 'min',
        every_n_epochs=1,
        save_last=True,
        verbose=True
    )
    
    # Callback to clean up empty dirs left by ModelCheckpoint
    class CleanEmptyDirs(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            ckpt_dir = checkpoint_callback.dirpath
            if ckpt_dir and os.path.isdir(ckpt_dir):
                for d in os.listdir(ckpt_dir):
                    p = os.path.join(ckpt_dir, d)
                    if os.path.isdir(p) and not os.listdir(p):
                        os.rmdir(p)

    # Create callbacks list
    callbacks = [checkpoint_callback, CleanEmptyDirs()]
    
    # Only add early stopping if patience is not -1
    if args.patience != -1:
        early_stopping_callback = EarlyStopping(
            monitor=args.monitor_loss,
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping_callback)
        print(f"Early stopping enabled with patience={args.patience}")
    else:
        print("Early stopping disabled (patience=-1)")

    # Load trainer config from config file if available, otherwise use defaults
    if hasattr(config, 'trainer') and config.trainer is not None:
        trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
        print("Loading trainer configuration from config file:")
        print(OmegaConf.to_yaml(config.trainer))
    else:
        # Default trainer config
        trainer_config = {
            "max_epochs": 100,
            "accelerator": "gpu",
            "devices": 1,
            "precision": "bf16",
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 10,
            "deterministic": True,
        }
        print("Using default trainer configuration")

    
    # Always override callbacks and logger from runtime
    trainer_config["callbacks"] = callbacks
    trainer_config["logger"] = logger
    
    trainer = Trainer(**trainer_config)

    if args.batch_size != -1:
        dm.hparams.batch_size = args.batch_size

    model.learning_rate = config.model.base_learning_rate

    if args.train:
        model.train()
        # Use Lightning's built-in resume mechanism which restores:
        # - Model weights
        # - Optimizer state
        # - Learning rate schedulers
        # - Epoch/step counters
        # - RNG states

        if resume_from_checkpoint:
            model.init_from_ckpt(resume_from_checkpoint)
        trainer.fit(model, dm)

    if args.ckpt == "":
        args.ckpt = get_best_checkpoint(args.project)
        
    if args.test:
        print(f"Loading checkpoint from {args.ckpt}")
        dm.setup('test')
        model.init_from_ckpt(path=args.ckpt)
        if isinstance(model, DDPM):
            loss = test_lpips_ldm(dm.test_dataloader(), model)
        else:
            loss = test_lpips(dm.test_dataloader(), model)
        print(f'LPIPS loss: {loss}')
        
        # Log test results to wandb
        if wandb.run is not None:
            wandb.log({"test/lpips_loss": loss})

    if args.sample:
        print(f"Loading checkpoint from {args.ckpt}")
        dm.setup()
        model.init_from_ckpt(path=args.ckpt)

        model.eval()
        if args.sample_from_train:
            test_loader = dm.train_dataloader()
        else:
            test_loader = dm.test_dataloader()

        batch = next(iter(test_loader))

        if isinstance(model, DDPM):

            (orig_cond_img, ehr_c), (orig_gt_img, ehr_gt), label, diff = batch

            out = model.get_input(batch)
            cond_img = out['cond_z']
            cond_ehr = out['cond_ehr']
            gt_img = out['gt_z']
            gt_ehr = out['gt_ehr']
            batched_prompts = out['prompts']

            cond_img = cond_img.to(device).to(model.dtype)
            cond_ehr = cond_ehr.to(device).to(model.dtype)
            gt_img = gt_img.to(device).to(model.dtype)

            sampler = DDIMSampler(model)
            sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=True)

            ddim_start = torch.randn(args.batch_size, 4, 8, 8).to(device).to(model.dtype)
            c = model.get_learned_conditioning(batched_prompts).to(device).to(model.dtype)

            samples_ddim = sampler.decode(
                ddim_start,
                cond=c,
                t_start=49,
                sc = None,
                c_ehr=cond_ehr,
                c_ref=cond_img,
                gt_ehr=gt_ehr,
                # unconditional_guidance_scale=2.0,
                # unconditional_conditioning=1,
            )

            samples_ddim = model.decode_first_stage(samples_ddim)

            # Print indices where label equals 1
            if isinstance(label, torch.Tensor):
                # Convert to numpy if it's a tensor
                label_np = label.cpu().numpy()
            else:
                label_np = label
            
            # Find indices where value is 1
            indices_of_ones = (label_np == 1).nonzero()[0] if label_np.ndim == 1 else (label_np == 1).nonzero()
            print(f"Indices of 1s in label: {indices_of_ones}")
            print(f"Original label: {label}")

            sample_img([orig_cond_img, samples_ddim, orig_gt_img], legend=['Condition', 'Reconstruction', 'Ground Truth'], save_path=f'samples/ldm/{args.project}/{"train" if args.sample_from_train else "test"}', labels=label)
        else:
            with torch.no_grad():
                autoencoder_batch = model.get_input(batch)

                if len(autoencoder_batch) == 2:
                    cond_img, label = autoencoder_batch
                elif len(autoencoder_batch) == 3:
                    cond_img, feature, label = autoencoder_batch
                else:
                    raise NotImplementedError

                posterior = model.encode(cond_img)

                # Generate samples dynamically based on sample_count
                samples = [cond_img]  # Start with condition image
                legend = ['Condition']
                
                for i in range(args.sample_count):
                    if i == 0:
                        # First sample uses mode without noise
                        z = posterior.sample()
                    else:
                        # Additional samples add noise
                        z = posterior.mode() + torch.randn_like(posterior.mode())

                    legend.append(f'Reconstruction {i + 1}')
                    
                    dec = model.decode(z)
                    samples.append(dec)

                sample_img(samples, legend=legend, save_path=f'samples/autoencoder/{args.project}/{"train" if args.sample_from_train else "test"}', greyscale=True)


if __name__ == "__main__":
    main()

