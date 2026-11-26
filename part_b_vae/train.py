"""Train the Part B VAE for 30 epochs and save the three figures.

Adam, learning rate 1e-3, batch size 128. Labels are not used in the loss.
Run from the repo root:

    python part_b_vae/train.py
"""
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

from uhat import load_uhat_characters
from viz import save_grid
from model import VAE, vae_loss

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
SEED = 0
DATA = ROOT / "starter" / "data" / "uhat"
RESULTS = ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "vae.pt"


def run_epoch(model, data, opt):
    """One pass over `data` (already on the device). Shuffled when training."""
    training = opt is not None
    model.train(training)
    totals = torch.zeros(3, device=data.device)
    if training:
        order = torch.randperm(len(data), device=data.device)
    else:
        order = torch.arange(len(data), device=data.device)
    for start in range(0, len(data), BATCH_SIZE):
        batch = data[order[start:start + BATCH_SIZE]]
        logits, mu, logvar = model(batch)
        recon, kl, total = vae_loss(logits, batch, mu, logvar)
        if training:
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
        totals += torch.stack([recon, kl, total]).detach() * batch.size(0)
    return (totals / len(data)).tolist()


def reconstructions(model, images, device):
    """Redraw each image from the centre mu of its code cloud."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(images.to(device))
        logits = model.decoder(mu)
    return torch.sigmoid(logits).cpu()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("device:", device)

    train_x, _, names = load_uhat_characters(root=DATA, split="train")
    test_x, test_y, _ = load_uhat_characters(root=DATA, split="test", class_names=names)
    print("UHaT train", tuple(train_x.shape), "test", tuple(test_x.shape), "classes", len(names))
    train_gpu = train_x.to(device)
    test_gpu = test_x.to(device)

    model = VAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("VAE parameters:", n_params)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_b_loss.csv"
    history = []
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,train_recon,train_kl,train_total,test_recon,test_kl,test_total\n")
        for epoch in range(1, EPOCHS + 1):
            train_stats = run_epoch(model, train_gpu, opt)
            with torch.no_grad():
                test_stats = run_epoch(model, test_gpu, None)
            row = [epoch, *train_stats, *test_stats]
            history.append(row)
            log.write(",".join(f"{v:.6f}" if i else str(v) for i, v in enumerate(row)) + "\n")
            log.flush()
            print(
                f"epoch {epoch:2d}  train {train_stats[2]:.1f} "
                f"(recon {train_stats[0]:.1f} kl {train_stats[1]:.2f})  "
                f"test {test_stats[2]:.1f}",
                flush=True,
            )
            # Save every epoch so a runtime disconnect does not lose the run.
            torch.save(model.state_dict(), CKPT)

    minutes = (time.time() - start) / 60
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    final = history[-1]
    (RESULTS / "part_b_training.txt").write_text(
        f"parameters: {n_params}\nfinal_loss: {final[3]:.4f}\n"
        f"minutes: {minutes:.1f}\ndevice: {gpu}\n"
        f"test_neg_elbo: {final[6]:.4f}\ntest_recon: {final[4]:.4f}\ntest_kl: {final[5]:.4f}\n",
        encoding="utf-8",
    )
    print(f"test negative ELBO {final[6]:.2f} = recon {final[4]:.2f} + KL {final[5]:.2f} nats")
    print("saved", CKPT)

    cols = list(zip(*history))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, title, train_col, test_col in zip(
        axes, ["reconstruction", "KL", "total"], (1, 2, 3), (4, 5, 6)
    ):
        ax.plot(cols[0], cols[train_col], label="train")
        ax.plot(cols[0], cols[test_col], label="test")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "part_b_loss.png", dpi=120)
    plt.close(fig)

    model.eval()
    step = max(len(test_x) // 16, 1)
    pick = torch.arange(0, step * 16, step)[:16]
    shown = test_x[pick]
    recon = reconstructions(model, shown, device)
    save_grid(torch.cat([shown, recon]), RESULTS / "part_b_reconstructions.png", nrow=8)
    print("saved reconstructions (two rows of originals, then two rows of redraws)")

    with torch.no_grad():
        samples = torch.sigmoid(model.decoder(torch.randn(64, 16, device=device))).cpu()
    save_grid(samples, RESULTS / "part_b_samples.png", nrow=8)
    print("saved 64 samples from N(0, I)")

    first = 0
    other = (test_y != test_y[0]).nonzero(as_tuple=True)[0][0].item()
    pair = test_x[[first, other]]
    with torch.no_grad():
        mu, _ = model.encoder(pair.to(device))
        weights = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
        z = (1 - weights) * mu[0] + weights * mu[1]
        morph = torch.sigmoid(model.decoder(z)).cpu()
    save_grid(morph, RESULTS / "part_b_interpolation.png", nrow=10)
    print(
        "saved interpolation:",
        names[int(test_y[first])],
        "to",
        names[int(test_y[other])],
    )


if __name__ == "__main__":
    main()
