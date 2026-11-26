"""Train the Part C diffusion model and save the three figures.

The handout fixes the noise schedule and the sampler. It does not fix the
optimiser, so this uses Adam at 2e-4 (the usual small-image diffusion rate),
batch size 128, and 30 epochs. Change the constants below if your handout
gives different numbers.

Run from the repo root:

    python part_c_diffusion/train.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uhat import load_uhat_characters
from viz import save_grid
from diffusion import Schedule, make_unet, noise_loss, noisy_image, sample

EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
SEED = 0
DATA = ROOT / "starter" / "data" / "uhat"
RESULTS = ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "unet.pt"
NOISE_TIMES = (1, 100, 200, 400, 600, 800, 900, 1000)
TRAJECTORY_TIMES = (1000, 800, 600, 400, 200, 100, 50, 0)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_x, train_y, names = load_uhat_characters(root=DATA, split="train")
    test_x, test_y, _ = load_uhat_characters(root=DATA, split="test", class_names=names)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    schedule = Schedule().to(device)
    print("alpha_bar at the last step:", float(schedule.alpha_bar[-1]))

    model = make_unet(num_classes=len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_c_loss.csv"
    history = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,loss\n")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total, count = 0.0, 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                loss = noise_loss(model, images, labels, schedule)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * images.shape[0]
                count += images.shape[0]
            epoch_loss = total / count
            history.append(epoch_loss)
            log.write(f"{epoch},{epoch_loss:.6f}\n")
            log.flush()
            print(f"epoch {epoch:2d}  loss {epoch_loss:.4f}", flush=True)

    torch.save(model.state_dict(), CKPT)
    print("saved", CKPT)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, EPOCHS + 1), history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("noise MSE")
    ax.set_title("Part C training loss")
    fig.tight_layout()
    fig.savefig(RESULTS / "part_c_loss.png", dpi=120)
    plt.close(fig)

    model.eval()
    clean = test_x[0:1].to(device)
    eps = torch.randn_like(clean)
    frames = []
    for t in NOISE_TIMES:
        tt = torch.full((1,), t, device=device)
        frames.append(noisy_image(clean, tt, eps, schedule).cpu())
    save_grid(
        torch.cat(frames),
        RESULTS / "part_c_forward_noise.png",
        nrow=8,
        titles=[f"t={t}" for t in NOISE_TIMES],
    )
    print("saved forward noising")

    per_class = []
    for _ in range(8):
        labels = torch.arange(len(names), device=device)
        drawn, _ = sample(model, labels, schedule)
        per_class.append(drawn.cpu())
    # per_class[k][c] is draw k of class c. Each row of the grid is one class.
    grid = torch.stack(per_class).permute(1, 0, 2, 3, 4).reshape(-1, 1, 28, 28)
    save_grid(grid, RESULTS / "part_c_samples.png", nrow=8)
    print("saved 8 samples of each character")

    letter = int(test_y[0])
    _, trajectory = sample(
        model,
        torch.tensor([letter], device=device),
        schedule,
        capture=TRAJECTORY_TIMES,
    )
    save_grid(
        torch.cat(trajectory),
        RESULTS / "part_c_trajectory.png",
        nrow=8,
        titles=[f"t={t}" for t in TRAJECTORY_TIMES],
    )
    print("saved denoising trajectory for", names[letter])


if __name__ == "__main__":
    main()
