"""A.3–A.4 Few-label linear classifier on the frozen encoder.

AdamW, learning rate 1e-3, 200 epochs, all 160 images in one batch.
The brief sets the learning rate and does not set weight decay, so AdamW
keeps its default of 0.01. Run after pretrain.py, from the repo root:

    python part_a_mae/classify.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eurosat import CLASS_NAMES, few_label_indices, load_eurosat_split, patchify
from model import LinearClassifier, MaskedAutoencoder, NUM_CLASSES

EPOCHS = 200
LR = 1e-3
SEED = 0
BATCH = 256

RESULTS = ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "mae.pt"
HEAD_CKPT = CKPT.parent / "classifier_head.pt"  # the trained Linear(192, 10), for the app


def confusion(pred, target, n=NUM_CLASSES):
    idx = target * n + pred
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def main():
    if not CKPT.is_file():
        raise SystemExit(f"Missing {CKPT}. Run part_a_mae/pretrain.py first.")
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_images, train_labels = load_eurosat_split("train")
    few = few_label_indices(train_labels)
    train_x = patchify(train_images[few]).contiguous().to(device)
    train_y = train_labels[few].to(device)

    mae = MaskedAutoencoder().to(device)
    mae.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    clf = LinearClassifier(mae).to(device)
    opt = torch.optim.AdamW(clf.head.parameters(), lr=LR)

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "part_a_classifier_loss.csv"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,loss,train_accuracy\n")
        for epoch in range(1, EPOCHS + 1):
            clf.train()
            logits = clf(train_x)
            loss = F.cross_entropy(logits, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc = (logits.argmax(1) == train_y).float().mean()
            log.write(f"{epoch},{loss.item():.6f},{acc.item():.6f}\n")
            if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
                print(f"epoch {epoch:3d}  loss {loss.item():.4f}  train acc {acc.item():.3f}", flush=True)

    torch.save(clf.head.state_dict(), HEAD_CKPT)
    print("saved", HEAD_CKPT)

    test_images, test_labels = load_eurosat_split("test")
    test_x = patchify(test_images).contiguous()
    clf.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_x), BATCH):
            logits = clf(test_x[start:start + BATCH].to(device))
            preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds)
    accuracy = (pred == test_labels).float().mean().item()
    cm = confusion(pred, test_labels)

    (RESULTS / "part_a_test_accuracy.txt").write_text(
        f"test_accuracy: {accuracy:.6f}\nimages: {len(test_labels)}\n",
        encoding="utf-8",
    )
    lines = [",".join(CLASS_NAMES)]
    for i, row in enumerate(cm.tolist()):
        lines.append(CLASS_NAMES[i] + "," + ",".join(str(int(v)) for v in row))
    (RESULTS / "part_a_confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Part A test accuracy {accuracy:.3f}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "part_a_confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"test accuracy: {accuracy:.4f}")
    print("saved", RESULTS / "part_a_confusion_matrix.png")


if __name__ == "__main__":
    main()
