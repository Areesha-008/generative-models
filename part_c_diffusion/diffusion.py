"""Part C diffusion: the noise schedule, one training step, and the sampler.

The U-Net comes from diffusers.UNet2DModel with the exact constructor in the
brief and random weights. No diffusers scheduler or pipeline is used.

Images live in [-1, 1]. Time t runs from 1 to T. t = 0 means the clean image.
"""
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

T = 1000
BETA_START = 1e-4
BETA_END = 0.02


class Schedule:
    """Linear beta from 1e-4 to 0.02. Tensor index t - 1 holds time t."""

    def __init__(self, steps=T, beta_start=BETA_START, beta_end=BETA_END):
        self.beta = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @property
    def steps(self):
        return self.beta.shape[0]

    def alpha_bar_at(self, t):
        """alpha_bar for a batch of times t in 0..T. alpha_bar at t = 0 is 1."""
        padded = torch.cat([torch.ones(1, device=self.alpha_bar.device), self.alpha_bar])
        return padded[t]

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def make_unet(num_classes=40):
    """The U-Net exactly as given in the brief."""
    return UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=num_classes,
    )


def noisy_image(x0, t, eps, schedule):
    """Closed-form forward process. t is a batch of integers from 0 to T."""
    ab = schedule.alpha_bar_at(t).view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def noise_loss(model, x0, labels, schedule):
    """L_simple: t ~ Uniform{1..T}, predict the noise, mean squared error."""
    t = torch.randint(1, schedule.steps + 1, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    xt = noisy_image(x0, t, eps, schedule)
    pred = model(xt, t, class_labels=labels).sample
    return F.mse_loss(pred, eps)


def denoise_step(model, x, t, labels, schedule):
    """DDPM Algorithm 2, one step from x_t to x_{t-1}. sigma_t^2 = beta_t, z = 0 at t = 1."""
    tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    eps = model(x, tt, class_labels=labels).sample
    beta = schedule.beta[t - 1]
    alpha = schedule.alpha[t - 1]
    alpha_bar = schedule.alpha_bar[t - 1]
    mean = (x - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
    if t == 1:
        return mean
    return mean + beta.sqrt() * torch.randn_like(x)


@torch.no_grad()
def sample(model, labels, schedule, capture=()):
    """Draw one character per label, starting from x_T ~ N(0, I).

    capture lists times to keep: t is the image x_t before the step at t,
    and 0 is the finished image. The result is clipped to [-1, 1].
    """
    model.eval()
    capture = set(capture)
    frames = {}
    x = torch.randn(labels.shape[0], 1, 28, 28, device=labels.device)
    for t in range(schedule.steps, 0, -1):
        if t in capture:
            frames[t] = x.clamp(-1, 1).cpu()
        x = denoise_step(model, x, t, labels, schedule)
    x = x.clamp(-1, 1)
    if 0 in capture:
        frames[0] = x.cpu()
    return x, frames


def to_unit(x):
    """[-1, 1] -> [0, 1] for display."""
    return (x + 1.0) / 2.0


if __name__ == "__main__":
    schedule = Schedule()
    last = schedule.alpha_bar[-1].item()
    assert 2e-5 < last < 8e-5, last
    assert schedule.alpha_bar_at(torch.tensor([0])).item() == 1.0

    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    eps = torch.randn_like(x0)
    assert torch.equal(noisy_image(x0, torch.tensor([0, 0]), eps, schedule), x0)
    xt = noisy_image(x0, torch.tensor([1, T]), eps, schedule)
    # Near the original at t = 1, almost pure noise at t = T.
    assert (xt[0] - x0[0]).abs().mean() < 0.05
    assert torch.allclose(xt[1], eps[1], atol=0.02)

    labels = torch.tensor([0, 3])
    model = make_unet()
    assert tuple(model.config.block_out_channels) == (32, 64, 128)
    loss = noise_loss(model, x0, labels, schedule)
    assert loss.ndim == 0

    step = denoise_step(model, xt, 1, labels, schedule)
    assert step.shape == x0.shape
    print("checks passed")
    print("alpha_bar_T:", last)
    print("U-Net parameters:", sum(p.numel() for p in model.parameters()))
    print("noise loss:", float(loss.detach()))
