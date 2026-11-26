"""A.1–A.2 Pretrain the masked autoencoder and save 8 reconstructions.

AdamW, learning rate 5e-4, weight decay 0.05, batch size 256, 100 epochs.
Labels are ignored. Run from the repo root:

    python part_a_mae/pretrain.py
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

from eurosat import CLASS_NAMES, load_eurosat_split, patchify, unpatchify
from model import MaskedAutoencoder, masked_mse

EPOCHS = 100
BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 0.05
SEED = 0

RESULTS = ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "mae.pt"


def epoch_loss(model, loader, opt, device):
    model.train()
    total = 0.0
    count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        pred, mask = model(batch)
        loss = masked_mse(pred, batch, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * batch.shape[0]
        count += batch.shape[0]
    return total / count


def save_reconstructions(model, images, labels, device, path):
    """Three rows: masked input (hidden patches grey), reconstruction, original."""
    model.eval()
    patches = patchify(images).to(device)
    with torch.no_grad():
        pred, mask = model(patches)
    pred = pred.cpu()
    mask = mask.cpu()
    patches = patches.cpu()
    hidden = mask.unsqueeze(-1).bool()
    grey = torch.full_like(patches, 0.5)
    masked_input = unpatchify(torch.where(hidden, grey, patches))
    pasted = torch.where(hidden, pred, patches).clamp(0, 1)
    reconstruction = unpatchify(pasted)
    original = images.cpu()
    rows = [masked_input, reconstruction, original]
    row_names = ["masked", "reconstruction", "original"]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5.2))
    for r, (row, name) in enumerate(zip(rows, row_names)):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(row[c].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(CLASS_NAMES[int(labels[c])], fontsize=7)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    images, _ = load_eurosat_split("train")
    patches = patchify(images).contiguous()
    loader = DataLoader(TensorDataset(patches), batch_size=BATCH_SIZE, shuffle=True)
    model = MaskedAutoencoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_a_pretrain_loss.csv"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,loss\n")
        for epoch in range(1, EPOCHS + 1):
            loss = epoch_loss(model, loader, opt, device)
            log.write(f"{epoch},{loss:.6f}\n")
            log.flush()
            print(f"epoch {epoch:3d}  loss {loss:.4f}", flush=True)

    torch.save(model.state_dict(), CKPT)
    print("saved", CKPT)

    losses = [float(line.split(",")[1]) for line in log_path.read_text(encoding="utf-8").splitlines()[1:]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.set_title("Part A pretraining loss")
    fig.tight_layout()
    fig.savefig(RESULTS / "part_a_pretrain_loss.png", dpi=120)
    plt.close(fig)

    test_images, test_labels = load_eurosat_split("test")
    gen = torch.Generator().manual_seed(SEED)
    pick = torch.randperm(len(test_images), generator=gen)[:8]
    save_reconstructions(
        model,
        test_images[pick],
        test_labels[pick],
        device,
        RESULTS / "part_a_reconstructions.png",
    )
    print("saved", RESULTS / "part_a_reconstructions.png")


if __name__ == "__main__":
    main()
