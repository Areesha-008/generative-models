"""Part B variational autoencoder. Latent code z has 16 numbers.

The encoder turns a 28x28 character into a Gaussian: a centre mu and a width
log sigma^2. One code is drawn with z = mu + sigma * noise. The decoder turns
that code back into one logit per pixel.
"""
import torch
import torch.nn.functional as F
from torch import nn

LATENT = 16
FLAT = 64 * 7 * 7  # 3136


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.mu = nn.Linear(FLAT, LATENT)
        self.logvar = nn.Linear(FLAT, LATENT)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, FLAT)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        x = torch.relu(self.deconv1(x))
        return self.deconv2(x)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)), averaged over the batch. Never negative.

    Zero when mu is 0 and log sigma^2 is 0, because that Gaussian is N(0, I).
    """
    per_image = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_image.mean()


def vae_loss(logits, x, mu, logvar):
    """Negative ELBO, averaged over the batch.

    Reconstruction is binary cross-entropy summed over the 784 pixels.
    """
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon = recon.view(x.size(0), -1).sum(dim=1).mean()
    kl = kl_divergence(mu, logvar)
    return recon, kl, recon + kl


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VAE()
    images = torch.rand(4, 1, 28, 28)
    logits, mu, logvar = model(images)
    assert logits.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 16) and logvar.shape == (4, 16)

    zeros = torch.zeros(8, 16)
    assert torch.allclose(kl_divergence(zeros, zeros), torch.tensor(0.0))
    assert kl_divergence(torch.randn(32, 16), torch.randn(32, 16)) >= 0

    recon, kl, total = vae_loss(logits, images, mu, logvar)
    assert torch.allclose(total, recon + kl)
    print("checks passed")
    print("logits", tuple(logits.shape), "mu", tuple(mu.shape))
    print("kl at N(0, I):", float(kl_divergence(zeros, zeros)))
