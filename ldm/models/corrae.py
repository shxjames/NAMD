import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config

class ContrastiveAutoencoderKL(AutoencoderKL):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 contrastiveconfig, 
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 label_loss_weight=None,
                 feature_loss_weight=None,
                 hidden_dim=128,
                 num_classes=1,
                 feature_dim=3,
                 weight_decay=0.0,
                 **contrastive_loss_kwargs,
               ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels, 
            monitor=monitor
        )

        self.contrastive_feature_loss = instantiate_from_config(contrastiveconfig)
        self.feature_loss_weight = feature_loss_weight

        self.label_loss = nn.BCEWithLogitsLoss()
        self.label_loss_weight = label_loss_weight

        self.label_head = nn.Linear(hidden_dim, num_classes)
        self.weight_decay = weight_decay


    @torch.no_grad()
    def get_input(self, batch,):
        img, feature, label = batch

        img = img.to(memory_format=torch.contiguous_format).to(dtype=self.dtype)
        img = img.to(self.device)
        label = label.to(self.device).to(dtype=self.dtype)
        feature = feature.to(self.device).to(dtype=self.dtype)


        del batch
        return img, feature, label
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)
        return dec, posterior, z

    def compute_image_metrics(self, inputs, reconstructions):
        """Compute SSIM and PSNR metrics between inputs and reconstructions."""
        # Normalize to [0, 1] range for metric computation (assuming inputs are in [-1, 1])
        inputs_norm = (inputs + 1.0) / 2.0
        recon_norm = (reconstructions + 1.0) / 2.0

        # Clamp to valid range
        inputs_norm = torch.clamp(inputs_norm, 0.0, 1.0)
        recon_norm = torch.clamp(recon_norm, 0.0, 1.0)

        # Compute SSIM
        ssim_val = ssim(recon_norm, inputs_norm, data_range=1.0)

        # Compute PSNR
        psnr_val = psnr(recon_norm, inputs_norm, data_range=1.0)

        return ssim_val, psnr_val
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, feature, label = self.get_input(batch,)
        reconstructions, posterior, z = self(inputs)

        b = reconstructions.shape[0]

        # Compute and log image quality metrics
        with torch.no_grad():
            ssim_val, psnr_val = self.compute_image_metrics(inputs, reconstructions)
            self.log("train/ssim", ssim_val, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/psnr", psnr_val, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        z_flat = z.view(b, -1)
        pred = self.label_head(z_flat)

        pred_loss = self.label_loss(pred, label[:, None].float())

        # Compute feature loss only for samples without NaN values
        # Create mask for valid (non-NaN) features
        nan_mask = torch.isnan(feature)  # Shape: (batch, feature_dim)
        has_any_nan = nan_mask.any(dim=1)  # Shape: (batch,) - True if any feature is NaN
        valid_samples = ~has_any_nan  # Samples without any NaN

        self.log("valid_samples", valid_samples.sum(), logger=True, on_step=True)
        
        if valid_samples.sum() > 0:
            # Only compute loss for valid samples
            feature_loss = self.contrastive_feature_loss(z[valid_samples], feature[valid_samples])
            self.log("train/contrastive_feature_loss", feature_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            # All samples have NaN, don't compute feature loss
            feature_loss = torch.tensor(0.0, device=z.device)


        self.log("pred_loss", pred_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae, nll_loss = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", return_nll_loss=True)

            label_loss_weight = self.calculate_adaptive_weight(nll_loss, pred_loss) if self.label_loss_weight is None else self.label_loss_weight
            self.log("label_loss_weight", label_loss_weight, prog_bar=True, logger=True, on_step=True, on_epoch=True)


            if valid_samples.sum() > 0:
                feature_loss_weight = self.calculate_adaptive_weight(nll_loss, feature_loss) if self.feature_loss_weight is None else self.feature_loss_weight
                self.log("feature_loss_weight", feature_loss_weight, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            else:
                feature_loss_weight = torch.tensor(0.0, device=z.device)

            
            
            
            total_loss = aeloss + label_loss_weight * pred_loss + feature_loss_weight * feature_loss
            log_dict_ae["train/total_loss"] = total_loss
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

            output_dict = {
                "loss": total_loss,
                "target": label.detach(),
                "pred": pred.detach(),
                "pred_loss": pred_loss.detach(),
                "nll_loss": nll_loss.detach(),
            }
            if feature_loss > 0.0:
                output_dict["feature_loss"] = feature_loss.detach()

            return output_dict

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def get_last_encoder_layer(self):
        """Get the last layer of the encoder for gradient-based weight calculation."""
        return self.encoder.conv_out.weight
    
    def calculate_adaptive_weight(self, recon_loss, custom_loss,):
        recon_grads = torch.autograd.grad(recon_loss, self.get_last_encoder_layer(), retain_graph=True)[0]
        custom_grads = torch.autograd.grad(custom_loss, self.get_last_encoder_layer(), retain_graph=True)[0]

        weight = torch.norm(recon_grads) / (torch.norm(custom_grads) + 1e-4)
        weight = torch.clamp(weight, 0.0, 1e4).detach()
        
        return weight

        
    def validation_step(self, batch, batch_idx):
        inputs, feature, label = self.get_input(batch,)
        reconstructions, posterior, z = self(inputs)

        b = reconstructions.shape[0]

        # Compute and log image quality metrics
        ssim_val, psnr_val = self.compute_image_metrics(inputs, reconstructions)
        self.log("val/ssim", ssim_val, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr_val, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        z_flat = z.view(b, -1)
        pred = self.label_head(z_flat)

        pred_loss = self.label_loss(pred, label[:, None].float())
        self.log("val/pred_loss", pred_loss,)

        
        # Compute feature loss only for samples without NaN values
        nan_mask = torch.isnan(feature)
        has_any_nan = nan_mask.any(dim=1)
        valid_samples = ~has_any_nan
        
        if valid_samples.sum() > 0:
            feature_loss = self.contrastive_feature_loss(z[valid_samples], feature[valid_samples])
            self.log("val/contrastive_feature_loss", feature_loss,)
        else:
            feature_loss = torch.tensor(0.0, device=z.device)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/ae_loss", aeloss)
        self.log("val/discloss", discloss)
        
        feature_loss_weight = self.feature_loss_weight if self.feature_loss_weight is not None else 1.0
        label_loss_weight = self.label_loss_weight if self.label_loss_weight is not None else 1.0

        log_dict_ae["val/total_loss"] += feature_loss_weight * feature_loss + label_loss_weight * pred_loss
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        inputs, feature, label = self.get_input(batch)
        reconstructions, posterior, z = self(inputs)

        b = reconstructions.shape[0]

        # Compute and log image quality metrics
        ssim_val, psnr_val = self.compute_image_metrics(inputs, reconstructions)
        self.log("test/ssim", ssim_val, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("test/psnr", psnr_val, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        z = z.view(b, -1)
        pred = self.label_head(z)

        pred_loss = self.label_loss(pred, label[:, None].float())
        self.log("test/pred_loss", pred_loss,)

        
        # Compute feature loss only for samples without NaN values
        nan_mask = torch.isnan(feature)
        has_any_nan = nan_mask.any(dim=1)
        valid_samples = ~has_any_nan
        
        if valid_samples.sum() > 0:
            feature_loss = self.contrastive_feature_loss(z[valid_samples], feature[valid_samples])
            self.log("test/feature_loss", feature_loss,)
        else:
            feature_loss = torch.tensor(0.0, device=z.device)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="test")
        total_loss = aeloss + self.label_loss_weight * pred_loss + self.feature_loss_weight * feature_loss
        self.log("test/total_loss", total_loss,)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="test")
        self.log("test/rec_loss", log_dict_ae["test/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        
        # Core autoencoder parameters
        main_params = list(self.encoder.parameters()) + \
                     list(self.decoder.parameters()) + \
                     list(self.quant_conv.parameters()) + \
                     list(self.post_quant_conv.parameters())
        
        if self.feature_loss_weight is None or self.feature_loss_weight > 0:
            main_params += list(self.contrastive_feature_loss.parameters())
        if self.label_loss_weight is None or self.label_loss_weight > 0:
            main_params += list(self.label_head.parameters())
        
        opt_ae = torch.optim.AdamW(main_params, lr=lr, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def to_rgb(self, x):
        """Convert tensor to RGB format for visualization"""
        x = (x + 1.0) / 2.0 * 255.0
        x = x.to(torch.uint8)
        return x

    @torch.no_grad()
    def log_images(self, batch):
        """Generate images for logging"""
        log = dict()
        inputs, feature, label = self.get_input(batch)
        
        inputs = inputs.to(self.device)
        reconstructions, posterior, z = self(inputs)
        
        log["inputs"] = self.to_rgb(inputs)
        log["reconstructions"] = self.to_rgb(reconstructions)
        
        return log

    def on_train_epoch_end(self):
        """Log images every 5 epochs"""
        if self.current_epoch % 5 == 0:
            # Get a batch from validation loader
            val_loader = self.trainer.val_dataloaders
            if val_loader is not None:
                if isinstance(val_loader, list):
                    val_loader = val_loader[0]
                
                batch = next(iter(val_loader))
                log = self.log_images(batch)
                
                # Log to wandb if available
                if hasattr(self.trainer.logger, 'experiment') and self.trainer.logger.experiment is not None:
                    import wandb
                    if wandb.run is not None:
                        # Convert tensors to numpy and create wandb images
                        inputs_np = log["inputs"].cpu().numpy()
                        reconstructions_np = log["reconstructions"].cpu().numpy()
                        
                        # Log first few samples from the batch
                        num_samples = min(4, inputs_np.shape[0])
                        
                        import numpy as np
                        
                        for i in range(num_samples):
                            # Convert from CHW to HWC for wandb
                            if inputs_np.shape[1] == 1:  # Grayscale
                                input_img = inputs_np[i, 0]  # Remove channel dimension
                                recon_img = reconstructions_np[i, 0]
                            else:  # RGB
                                input_img = inputs_np[i].transpose(1, 2, 0)
                                recon_img = reconstructions_np[i].transpose(1, 2, 0)
                            
                            # Combine input and reconstruction horizontally
                            combined_img = np.concatenate([input_img, recon_img], axis=1)
                            
                            wandb.log({
                                f"val/input_vs_reconstruction_{i}": wandb.Image(combined_img, caption=f"Left: Input | Right: Reconstruction (Sample {i})")
                            }, step=self.global_step)
