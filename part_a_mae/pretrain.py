"""A.1–A.2 Pretrain the masked autoencoder and save 8 reconstructions.

AdamW, learning rate 5e-4, weight decay 0.05, batch size 256, 100 epochs.
Labels are ignored. Run from the repo root:

    python part_a_mae/pretrain.py

On a GPU the whole training set sits in GPU memory and the forward pass runs
in bfloat16 (the loss is computed in float32). Set AMP=0 to train in float32.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and os.environ.get("AMP", "1") != "0"
    return device, use_amp


def epoch_loss(model, patches, opt, device, use_amp):
    """One pass over the data, already on `device`, in a fresh random order."""
    model.train()
    total = 0.0
    order = torch.randperm(len(patches), device=device)
    for start in range(0, len(patches), BATCH_SIZE):
        batch = patches[order[start:start + BATCH_SIZE]]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred, mask = model(batch)
        loss = masked_mse(pred.float(), batch, mask.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.detach() * batch.shape[0]
    return (total / len(patches)).item()


def save_reconstructions(model, images, labels, device, path):
    """Three rows: masked input (hidden patches grey), reconstruction, original."""
    model.eval()
    patches = patchify(images).to(device)
    with torch.no_grad():
        pred, mask = model(patches)
    pred = pred.float().cpu()
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
    device, use_amp = setup_device()
    print("device:", device, "| bf16 autocast:", use_amp)
    images, _ = load_eurosat_split("train")
    patches = patchify(images).contiguous().to(device)
    print("training patches:", tuple(patches.shape))
    model = MaskedAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("MAE parameters (encoder + decoder):", n_params)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_a_pretrain_loss.csv"
    losses = []
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,loss\n")
        for epoch in range(1, EPOCHS + 1):
            loss = epoch_loss(model, patches, opt, device, use_amp)
            losses.append(loss)
            log.write(f"{epoch},{loss:.6f}\n")
            log.flush()
            # Save every epoch so a runtime disconnect does not lose the run.
            torch.save(model.state_dict(), CKPT)
            print(f"epoch {epoch:3d}  loss {loss:.5f}  ({time.time() - start:.0f}s)", flush=True)
    minutes = (time.time() - start) / 60
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    (RESULTS / "part_a_training.txt").write_text(
        f"parameters: {n_params}\nfinal_loss: {losses[-1]:.6f}\n"
        f"minutes: {minutes:.1f}\ndevice: {gpu}\n",
        encoding="utf-8",
    )
    print("saved", CKPT)
    del patches

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
