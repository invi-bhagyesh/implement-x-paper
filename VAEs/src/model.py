import torch
import torch.nn as nn
import torch.nn.functional as F

class Enocder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        p = F.relu(self.fc1(x))
        p = F.relu(self.fc21(p))
        return p
    
class Latent(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        mu = self.mu(p)
        logvar = self.logvar(p)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar) # logvar to std devation 
        eps = torch.randn_like(std)  # Random noise
        z = mu + eps * std  # Sample from the latent space
        
        return z, mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        p = F.relu(self.fc1(z))
        p = torch.sigmoid(self.fc2(p))  # Sigmoid activation for reconstruction
        return p

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = Encoder(input_size, hidden_size)
        self.latent = Latent(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        p = self.encoder(x)
        z, mu, logvar = self.latent(p)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar, z