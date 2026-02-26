import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDistanceLoss(nn.Module):
    """
    KL Divergence loss between feature distance distribution and latent distance distribution.
    
    Both distributions are treated as categorical distributions over pairwise relationships:
    - P: distribution from feature distances (teacher/target)
    - Q: distribution from latent distances (student/predicted)
    
    Loss = KL(P || Q) = sum(P * log(P / Q))
    
    This encourages the latent space to preserve the distance structure of the feature space.
    """
    
    def __init__(
        self,
        feature_sigma: float = 1.0,
        latent_sigma: float = 1.0,
        temperature: float = 1.0,
        eps: float = 1e-13,
        num_channels: int = 16,
        dim: int = 8,
        proj_mode: str = "pooling",
        distance_type: str = "rbf",  # "rbf", "cosine"
    ):
        """
        Args:
            proj_dim: Dimension of latent projection
            feature_sigma: RBF kernel bandwidth for feature distances
            latent_sigma: RBF kernel bandwidth for latent distances
            temperature: Temperature for softmax normalization
            eps: Small constant for numerical stability
            topk: If set, only consider top-k nearest neighbors
            num_channels: Number of channels in latent (for projection)
            dim: Spatial dimension of latent (for projection)
            proj_mode: "pooling" or "flatten" for latent projection
            distance_type: Type of distance metric ("rbf", "cosine")
            symmetric: Whether to use symmetric KL divergence
        """
        super().__init__()
        self.feature_sigma = feature_sigma
        self.latent_sigma = latent_sigma
        self.temperature = temperature
        self.eps = eps
        self.proj_mode = proj_mode
        self.distance_type = distance_type
    
    def forward(self, latents: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between feature and latent distance distributions.
        
        Args:
            latents: Latent representations (B, C, H, W)
            features: Feature vectors (B, D)
            
        Returns:
            KL divergence loss (scalar)
        """
        B = latents.shape[0]
        
        if B < 2:
            return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        
        # 1. Compute feature distance distribution P
        P = self._compute_distance_distribution(features, sigma=self.feature_sigma, is_latent=False)
        
        # 2. Project and compute latent distance distribution Q
        if self.proj_mode == "pooling":
            z = latents.mean(dim=(-1, -2))  # (B, C)
        elif self.proj_mode == "flatten":
            z = latents.view(B, -1)  # (B, C*H*W)
        
        # z = self.proj(z)  # (B, proj_dim)
        Q = self._compute_distance_distribution(z, sigma=self.latent_sigma, is_latent=True)
        
        # 4. Compute KL divergence
        loss = self._kl_divergence(P, Q)
        
        return loss
    
    def _compute_distance_distribution(
        self, 
        x: torch.Tensor, 
        sigma: float,
        is_latent: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise distance and convert to probability distribution.
        
        Args:
            x: Input tensor (B, D)
            sigma: RBF kernel bandwidth
            is_latent: Whether this is the latent (for potential normalization)
            
        Returns:
            Probability distribution (B, B) with diagonal zeroed
        """
        B = x.shape[0]
        
        if self.distance_type == "rbf":
            dist_sq = torch.cdist(x, x, p=2) ** 2
            logits = -dist_sq / (2 * sigma ** 2)
        elif self.distance_type == "cosine":
            x_norm = F.normalize(x, dim=1)
            logits = x_norm @ x_norm.T
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")


        logits = logits.clone()
        logits.fill_diagonal_(float('-inf'))

        logits = logits / self.temperature

        probs = F.softmax(logits, dim=1)
        
        # Ensure no zeros for numerical stability
        probs = probs + self.eps
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        return probs
    
    def _kl_divergence(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(P || Q) = sum(P * log(P / Q))
        
        Args:
            P: Target distribution (B, B)
            Q: Predicted distribution (B, B)
            
        Returns:
            Mean KL divergence (scalar)
        """

        kl = P * (torch.log(P + self.eps) - torch.log(Q + self.eps))
        
        # Sum over distribution dimension, mean over batch
        kl = kl.sum(dim=1).mean()
        
        return kl
    