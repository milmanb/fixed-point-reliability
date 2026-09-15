"""Convolutional denoising autoencoder (proposal, Sec. 3: fewer than 1M parameters).

Design choices:
  * a dense 64-unit bottleneck, so the network is a learned projector onto a
    low-dimensional set, comparable to the pca64 projector;
  * no normalization layers: the idempotence term applies the network to its own
    outputs, and batch statistics would mix corrupted and restored images;
  * SiLU activations, so f is smooth and Jacobian-based signals are well defined;
  * a sigmoid output, so restorations stay in [0, 1].
"""

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, width=32, latent=64):
        super().__init__()
        self.config = {"width": width, "latent": latent}
        wide = 2 * width
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.SiLU(),                  # 28x28
            nn.Conv2d(width, wide, 4, stride=2, padding=1), nn.SiLU(),     # 14x14
            nn.Conv2d(wide, wide, 4, stride=2, padding=1), nn.SiLU(),      # 7x7
            nn.Flatten(),
            nn.Linear(wide * 49, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, wide * 49), nn.SiLU(),
            nn.Unflatten(1, (wide, 7, 7)),
            nn.ConvTranspose2d(wide, wide, 4, stride=2, padding=1), nn.SiLU(),   # 14x14
            nn.ConvTranspose2d(wide, width, 4, stride=2, padding=1), nn.SiLU(),  # 28x28
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, y):
        return torch.sigmoid(self.decoder(self.encoder(y)))


class Restorer:
    """Evaluation adapter: runs a network on its own device in float32 and returns y's dtype and device.

    The casts are differentiable, so first-order signals (torch.func.jvp) pass through.
    """

    differentiable = True

    def __init__(self, net, name, device):
        self.net = net.to(device).eval()
        self.name = name
        self.device = torch.device(device)

    def __call__(self, y):
        return self.net(y.to(self.device, torch.float32)).to(y)


def save_checkpoint(path, net, train_args, history):
    torch.save({"model_config": net.config, "state_dict": net.state_dict(),
                "train_args": train_args, "history": history}, path)


def load_restorer(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = ConvAutoencoder(**checkpoint["model_config"])
    net.load_state_dict(checkpoint["state_dict"])
    return Restorer(net, name=path.stem, device=device), checkpoint
