import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt)

    ds = instantiate_from_config(config.data)
    ds.setup()

    train_loader = ds.train_dataloader()

    model.eval()
    model.to('cuda')

    # Welford's online algorithm for computing mean and variance
    n_elements = 0
    running_mean = 0.0
    running_m2 = 0.0  # Sum of squared differences from mean
    
    print("Computing latent statistics using Welford's algorithm...")

    for batch_idx, (img, _, _) in enumerate(train_loader):
        img = img.to('cuda').to(model.dtype)
        
        with torch.no_grad():
            posterior = model.encode(img)
            z = posterior.sample()
        
        # Move to CPU and flatten to 1D
        z_cpu = z.cpu().flatten()
        
        # Welford's algorithm: process each element
        for value in z_cpu:
            n_elements += 1
            delta = value.item() - running_mean
            running_mean += delta / n_elements
            delta2 = value.item() - running_mean
            running_m2 += delta * delta2
        
        # Batch statistics for monitoring
        batch_mean = z.mean().item()
        batch_std = z.std().item()
        
        print(f"Batch {batch_idx}: mean={batch_mean:.4f}, std={batch_std:.4f}, running_mean={running_mean:.4f}, running_std={torch.sqrt(torch.tensor(running_m2 / n_elements)):.4f}")
        
        # Free memory
        del img, posterior, z, z_cpu
        torch.cuda.empty_cache()
        
        # Limit to first 100 batches for speed
        if batch_idx >= 100:
            break
    
    # Compute final statistics
    final_variance = running_m2 / n_elements
    final_std = final_variance ** 0.5
    
    print(f"\n{'='*60}")
    print(f"Overall Statistics (Welford's algorithm):")
    print(f"Total elements: {n_elements}")
    print(f"Mean: {running_mean:.6f}")
    print(f"Variance: {final_variance:.6f}")
    print(f"Std:  {final_std:.6f}")
    print(f"Recommended scale_factor: {1.0/final_std:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

