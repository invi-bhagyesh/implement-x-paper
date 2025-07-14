import torch
import torch.nn.functional as F 

def loss(inputs, targets, logvar, mu):
    """
    Args:
        inputs (Tensor): Reconstructed output from the decoder (after sigmoid), shape [batch_size, ...].
        targets (Tensor): Ground truth input data, same shape as inputs.
        logvar (Tensor): Log variance of the latent distribution (from encoder).
        mu (Tensor): Mean of the latent distribution (from encoder).

    Returns:
        Tensor: Scalar tensor representing the total VAE loss (BCE + KLD).
    """

    #Cross-entropy loss
    bce = F.binary_cross_entropy(inputs, targets, reduction="sum") #total loss over batch
    #KL divergence loss
    kld = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())

    return bce + kld