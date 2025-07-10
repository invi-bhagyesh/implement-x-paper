class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder (input: 1, 28, 28)
        self.encoder = nn.Sequential(
             nn.Flatten(),              # (1, 28, 28) -> (784,)
             nn.Linear(784, 256),       # Fully connected layer: 784 -> 256
             nn.ReLU(),                 # Activation
             nn.Linear(256, 64),         # Fully connected layer: 256 -> 64
             nn.ReLU(),                 # Activation
             nn.Linear(64, 16),         # Fully connected layer: 64 -> 16
             
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()           # Scale to 0 to 1 
        )
        
    def forward(self, x):
        encoded = self.encoder(x)         # compress input
        decoded = self.decoder(encoded)  # reconstruct
        return decoded
