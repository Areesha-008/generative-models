"""C.1–C.5 Train the class-conditional DDPM and save its figures.

Adam, learning rate 2e-4, batch size 128, 40 epochs. Pixels are rescaled
from the starter's [0, 1] to [-1, 1]. Run from the repo root:

    python part_c_diffusion/train.py
"""
import sys
import time
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
from diffusion import Schedule, make_unet, noise_loss, noisy_image, sample, to_unit

EPOCHS = 40
BATCH_SIZE = 128
LR = 2e-4
SEED = 0
SAMPLES_PER_CLASS = 8
DATA = ROOT / "starter" / "data" / "uhat"
RESULTS = ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "unet.pt"
NOISE_TIMES = (0, 100, 250, 500, 750, 999)
TRAJECTORY_TIMES = (999, 750, 500, 250, 100, 0)


def to_signed(x):
    """[0, 1] -> [-1, 1]."""
    return x * 2.0 - 1.0


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_x, train_y, names = load_uhat_characters(root=DATA, split="train")
    test_x, test_y, _ = load_uhat_characters(root=DATA, split="test", class_names=names)
    train_x, test_x = to_signed(train_x), to_signed(test_x)
    print("pixel range:", float(train_x.min()), float(train_x.max()))
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    schedule = Schedule().to(device)
    print("alpha_bar_T:", float(schedule.alpha_bar[-1]))

    model = make_unet(num_classes=len(names)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("U-Net parameters:", n_params)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_c_loss.csv"
    history = []
    start = time.time()
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
            # Save every epoch so a Colab/Kaggle disconnect does not lose the run.
            torch.save(model.state_dict(), CKPT)
            print(f"epoch {epoch:2d}  loss {epoch_loss:.4f}", flush=True)
    minutes = (time.time() - start) / 60
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    (RESULTS / "part_c_training.txt").write_text(
        f"parameters: {n_params}\nfinal_loss: {history[-1]:.6f}\n"
        f"minutes: {minutes:.1f}\ndevice: {gpu}\n",
        encoding="utf-8",
    )
    print("saved", CKPT)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, EPOCHS + 1), history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("L_simple (noise MSE)")
    ax.set_title("Part C training loss")
    fig.tight_layout()
    fig.savefig(RESULTS / "part_c_loss.png", dpi=120)
    plt.close(fig)

    # C.2 Forward process on one training image.
    clean = train_x[0:1].to(device)
    eps = torch.randn_like(clean)
    frames = [
        noisy_image(clean, torch.tensor([t], device=device), eps, schedule).cpu()
        for t in NOISE_TIMES
    ]
    save_grid(
        to_unit(torch.cat(frames)),
        RESULTS / "part_c_forward_noise.png",
        nrow=len(NOISE_TIMES),
        titles=[f"t={t}" for t in NOISE_TIMES],
    )
    print("saved forward noising of", names[int(train_y[0])])

    # C.5 8 samples of each of the 40 characters, one row per character.
    labels = torch.arange(len(names), device=device).repeat_interleave(SAMPLES_PER_CLASS)
    drawn, _ = sample(model, labels, schedule)
    titles = [names[int(c)] if i % SAMPLES_PER_CLASS == 0 else "" for i, c in enumerate(labels)]
    save_grid(
        to_unit(drawn),
        RESULTS / "part_c_samples.png",
        nrow=SAMPLES_PER_CLASS,
        titles=titles,
    )
    print(f"saved {SAMPLES_PER_CLASS} samples of each character")

    # C.5 Denoising of one sample.
    letter = int(test_y[0])
    _, trajectory = sample(
        model,
        torch.tensor([letter], device=device),
        schedule,
        capture=TRAJECTORY_TIMES,
    )
    save_grid(
        to_unit(torch.cat([trajectory[t] for t in TRAJECTORY_TIMES])),
        RESULTS / "part_c_trajectory.png",
        nrow=len(TRAJECTORY_TIMES),
        titles=[f"t={t}" for t in TRAJECTORY_TIMES],
    )
    print("saved denoising trajectory for", names[letter])


if __name__ == "__main__":
    main()
