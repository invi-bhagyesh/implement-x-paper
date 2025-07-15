# Variational Autoencoder (VAE) for MNIST

- Modular VAE implementation in PyTorch
- Configurable model and training parameters
- Checkpointing and model saving/loading
- Callback support for custom training hooks (e.g., plotting reconstructions)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/invi-bhagyesh/implement-x-paper
   cd implement-x-paper/VAEs
   ```
2. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

### Configuration

Edit `src/config.json` to change model or training parameters:

```json
{
    "epochs": 20,
    "batch_size": 64,
    "lr": 0.001,
    "save_path": "hs-512_ls-20.pt",
    "model_params": {
        "input_size": 784,
        "hidden_size": 512,
        "latent_size": 20
    }
}
```

### Model Architecture

- **Encoder**: Two fully connected layers with ReLU activations
- **Latent**: Computes mean and log-variance, samples latent vector using the reparameterization trick
- **Decoder**: Two fully connected layers with ReLU and Sigmoid activations

### Loss Function

The VAE uses a combination of binary cross-entropy (BCE) for reconstruction and Kullback-Leibler divergence (KLD) for regularization:

```
loss = BCE(reconstructed, target) + KLD(mu, logvar)
```

##  References

- **Kingma, D. P., & Welling, M. (2014)**  
  *Auto-Encoding Variational Bayes*.  
  [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

- **PyTorch Documentation**  
  https://pytorch.org/docs/stable/index.html

- **Torchvision MNIST Dataset**  
  https://pytorch.org/vision/stable/datasets.html#mnist

- **Lilian Weng’s Blog – VAE Explained**  
  https://lilianweng.github.io/posts/2018-08-12-vae/

- **bvezilic' github repository **
  https://github.com/bvezilic/Variational-autoencoder?tab=readme-ov-file



