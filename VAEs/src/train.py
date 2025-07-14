from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
from model import VAE
from loss import loss
from utils import save_checkpoint, load_checkpoint, save_model, load_model
import json
import argparse
from config import DATA_DIR, MODELS_DIR
import os

class Trainer:
    def __init__(self, model, dataloader, optim, device, callback=[]):
        self.model = model
        self.dataloader = dataloader
        self.optim = optim
        self.device = device
        self.callback = callback

    def train(self, epochs):
        self.model.to(self.device
        self.model.train())

        losses = []
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for inputs, _ in self.dataloader:
                self.optim.zero_grad()

                x = inputs.view(inputs.size(0), -1).to(self.device)
                y = (x> 0.5).float().to(self.device)  # Binarize inputs

                # Forward pass
                reconstructed_x, mu, logvar = self.model(x)

                # Compute loss
                batch_loss = loss(reconstructed_x, y, logvar, mu)
                
                batch_loss.backward()
                self.optim.step()

                running_loss += batch_loss

            epoch_loss = running_loss / len(self.dataloader)
            losses.append(epoch_loss.item())
            print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss.item()}")

            # Callbacks
            if self.callback:
                [fn(self) for fn in self.callback]
                
        return losses
    
def train():
    config = json.load(open(args.config, "r"))

    dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    if args.resume:
        model, optim = load_checkpoint(os.path.join(MODELS_DIR, args.resume))
    else:
        model = VAE(**config["model_params"])
        optim = Adam(model.parameters(), lr=config["lr"])

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optim=optim,
        device=args.device
        callbacks=[]
    )

    trainer.train(epochs=config["epochs"])
    
    save_checkpoint(model, optim, os.path.join(MODELS_DIR, config["save_path"]))
    save_model(model, os.path.join(MODELS_DIR, config["save_path"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder on MNIST")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint to resume training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    
    args = parser.parse_args()
    
    train()
    