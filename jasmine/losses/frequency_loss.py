
import jax
import jax.numpy as jnp
import flax.nnx as nnx

class FocalFrequencyLoss(nnx.Module):
    """
    Focal Frequency Loss (FFL) for Image Reconstruction and Synthesis (ICCV 2021).
    Paper: https://arxiv.org/abs/2012.12821
    
    This loss focuses on optimizing the frequency domain differences between 
    the predicted and target images, helping to recover high-frequency details.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        alpha (float): Scaling factor for the spectrum weight matrix. Default: 1.0
    """
    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0):
        self.loss_weight = loss_weight
        self.alpha = alpha

    def __call__(self, pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            pred (jnp.ndarray): Predicted images of shape (..., H, W, C)
            target (jnp.ndarray): Target images of shape (..., H, W, C)
            
        Returns:
            jnp.ndarray: The calculated loss scalar.
        """
        # We perform FFT over the spatial dimensions, assumed to be (-3, -2) 
        # given input layout (..., H, W, C).
        spatial_axes = (-3, -2)
        
        # 1. Compute 2D FFT with orthogonal normalization to preserve energy and scale
        pred_freq = jnp.fft.fft2(pred, axes=spatial_axes, norm="ortho")
        target_freq = jnp.fft.fft2(target, axes=spatial_axes, norm="ortho")
        
        # 2. Compute spectrum distance (amplitude difference)
        # We use the squared Euclidean distance in the frequency domain.
        diff = pred_freq - target_freq
        diff_sq = jnp.abs(diff) ** 2
        
        # 3. Dynamic Spectrum Weighting
        # The paper defines weight w(u,v) based on the distance itself to focus on hard frequencies.
        # w(u,v) = |F_p - F_t|^alpha
        # Loss = w(u,v) * |F_p - F_t|^2 = |F_p - F_t|^(2 + alpha)
        
        # To avoid numerical instability with 0s, we add epsilon if needed, 
        # but power function usually handles >= 0 fine.
        
        # matrix = diff_sq ** (self.alpha / 2.0) # |diff|^alpha
        # focal_loss = matrix * diff_sq          # |diff|^(2+alpha)
        
        focal_loss = diff_sq ** (1.0 + self.alpha / 2.0)
        
        # 4. Average over all dimensions
        return self.loss_weight * jnp.mean(focal_loss)
