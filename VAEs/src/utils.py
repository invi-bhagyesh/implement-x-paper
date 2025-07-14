import torch 
import os
import torch.nn as nn
from model import VAE

def save_checkpoint(model, optimizer, path):
    """
    Save the model and optimizer state to a file.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save({
        'model_state_dict': model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "model": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "latent_size": model.latent_size
        }
    }, path)

def load_checkpoint(path):
    """
    Load the model and optimizer state from a file.
    """
    checkpoint = torch.load(path)
    model = VAE(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = nn.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    return model, optimizer

def save_model(model, path):
    """
    Save the model architecture and weights to a file.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save({
        "model_state_dict": model.state_dict(),
        "model": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "latent_size": model.latent_size
        }
    }, path)


def load_model(path):
    """
    Load the model architecture and weights from a file.
    """
    restore_dict = torch.load(path)
    model = VAE(**restore_dict["model"])
    model.load_state_dict(restore_dict["model_state_dict"])
    model.eval()
    return model