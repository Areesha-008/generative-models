"""Part C diffusion: the noise schedule, one training step, and the sampler.

The U-Net itself comes from diffusers.UNet2DModel and starts with random
weights. No diffusers scheduler or pipeline is used.

The handout does not list the U-Net channel widths. make_unet is a small
28x28 network with one embedding per Urdu character. Replace that function
if the handout gives a different constructor.
"""
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

T = 1000
BETA_START = 1e-4
BETA_END = 0.02


class Schedule:
    """Linear beta from 1e-4 to 0.02. Index 0 is time t = 1."""

    def __init__(self, steps=T, beta_start=BETA_START, beta_end=BETA_END):
        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        # alpha_bar at t-1. For t = 1 this is 1, meaning a clean image.
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        return self


def make_unet(num_classes=40):
    return UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
        num_class_embeds=num_classes,
    )


def noisy_image(x0, t, eps, schedule):
    """Jump straight to time t. t is a batch of integers from 1 to T."""
    ab = schedule.alpha_bar[t - 1].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def noise_loss(model, x0, labels, schedule):
    """One lesson: guess the noise that was mixed into the image."""
    t = torch.randint(1, schedule.beta.shape[0] + 1, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    xt = noisy_image(x0, t, eps, schedule)
    pred = model(xt, t, class_labels=labels).sample
    return F.mse_loss(pred, eps)


def denoise_step(model, x, t, labels, schedule):
    """Remove one level of noise. t is an integer from T down to 1.

    No fresh noise is added when t is 1.
    """
    tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    eps = model(x, tt, class_labels=labels).sample
    beta = schedule.beta[t - 1]
    alpha = schedule.alpha[t - 1]
    alpha_bar = schedule.alpha_bar[t - 1]
    mean = (1.0 / alpha.sqrt()) * (x - beta / (1.0 - alpha_bar).sqrt() * eps)
    if t == 1:
        return mean
    # Posterior variance from the DDPM paper. It is 0 at t = 1 as well.
    alpha_bar_prev = schedule.alpha_bar_prev[t - 1]
    variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta
    return mean + variance.sqrt() * torch.randn_like(x)


@torch.no_grad()
def sample(model, labels, schedule, capture=()):
    """Draw characters. labels has one class id per image.

    capture lists clock times to keep. 1000 is the starting static.
    0 is the finished image, after the last step.
    """
    model.eval()
    capture = set(capture)
    frames = []
    x = torch.randn(labels.shape[0], 1, 28, 28, device=labels.device)
    for t in range(schedule.beta.shape[0], 0, -1):
        if t in capture:
            frames.append(x.detach().cpu().clone())
        x = denoise_step(model, x, t, labels, schedule)
    if 0 in capture:
        frames.append(x.detach().cpu().clone())
    return x, frames


if __name__ == "__main__":
    schedule = Schedule()
    last = schedule.alpha_bar[-1].item()
    assert 2e-5 < last < 8e-5, last

    x0 = torch.rand(2, 1, 28, 28)
    t = torch.tensor([1, T])
    eps = torch.randn_like(x0)
    xt = noisy_image(x0, t, eps, schedule)
    assert xt.shape == x0.shape
    # At t = 1 the page stays near the original. At t = T it has moved far away.
    assert (xt[0] - x0[0]).abs().mean() < (xt[1] - x0[1]).abs().mean()

    labels = torch.tensor([0, 3])
    model = make_unet()
    loss = noise_loss(model, x0, labels, schedule)
    assert loss.ndim == 0

    step = denoise_step(model, xt, 1, labels, schedule)
    assert step.shape == x0.shape
    print("checks passed")
    print("alpha_bar at the last step:", last)
    print("noise loss:", float(loss.detach()))
