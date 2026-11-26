"""§3.3 Blind recognisability test for the Part C diffusion model.

Step 1, on the GPU after train.py (from the repo root):

    python part_c_diffusion/blind_test.py generate

This draws one sample of each of the 40 characters, shuffles them, and writes:
  results/part_c_blind_test.png        40 numbered images, no labels
  results/part_c_blind_test_names.txt  the 40 valid character names
  results/blind_member1.csv            sheet for member 1 (image, guess)
  results/blind_member2.csv            sheet for member 2 (image, guess)
  results/part_c_blind_key.csv         the answers. Do not open it until both sheets are done.

Step 2: each member fills in the guess column of their own sheet alone,
using names from part_c_blind_test_names.txt.

Step 3, anywhere (no GPU or torch needed):

    python part_c_diffusion/blind_test.py score

This writes the diffusion rows of results/human_eval.csv (other rows are kept)
and the Table 3 numbers to results/part_c_blind_test_summary.txt.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "starter" / "data" / "uhat"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "unet.pt"
SEED = 1234

IMAGE = RESULTS / "part_c_blind_test.png"
NAMES = RESULTS / "part_c_blind_test_names.txt"
KEY = RESULTS / "part_c_blind_key.csv"
SHEETS = [RESULTS / "blind_member1.csv", RESULTS / "blind_member2.csv"]
HUMAN_EVAL = RESULTS / "human_eval.csv"
SUMMARY = RESULTS / "part_c_blind_test_summary.txt"
HUMAN_EVAL_FIELDS = ["task", "item", "reference", "member1", "member2"]
TASK = "diffusion_blind"


def generate():
    import torch

    sys.path.insert(0, str(ROOT / "starter"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from uhat import load_uhat_characters
    from viz import save_grid
    from diffusion import Schedule, make_unet, sample, to_unit

    if not CKPT.is_file():
        raise SystemExit(f"Missing {CKPT}. Run part_c_diffusion/train.py first.")
    for sheet in SHEETS:
        if sheet.is_file() and any(row["guess"].strip() for row in _read(sheet)):
            raise SystemExit(f"{sheet} already has guesses. Delete it to start a new test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, names = load_uhat_characters(root=DATA, split="train")
    model = make_unet(num_classes=len(names)).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    schedule = Schedule().to(device)

    torch.manual_seed(SEED)
    labels = torch.arange(len(names), device=device)
    drawn, _ = sample(model, labels, schedule)
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(SEED))

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_grid(
        to_unit(drawn.cpu()[order]),
        IMAGE,
        nrow=8,
        titles=[str(i + 1) for i in range(len(names))],
    )
    NAMES.write_text("\n".join(names) + "\n", encoding="utf-8")
    _write(KEY, ["image", "character"],
           [{"image": i + 1, "character": names[int(c)]} for i, c in enumerate(order)])
    for sheet in SHEETS:
        _write(sheet, ["image", "guess"],
               [{"image": i + 1, "guess": ""} for i in range(len(names))])
    print("saved", IMAGE)
    print("each member fills in", ", ".join(s.name for s in SHEETS), "alone")


def score():
    names = {n.strip().lower() for n in NAMES.read_text(encoding="utf-8").split()}
    key = {row["image"]: row["character"].lower() for row in _read(KEY)}
    guesses = []
    for sheet in SHEETS:
        rows = {row["image"]: row["guess"].strip().lower() for row in _read(sheet)}
        missing = [i for i in key if not rows.get(i)]
        unknown = sorted({g for g in rows.values() if g and g not in names})
        if missing:
            raise SystemExit(f"{sheet.name}: no guess for images {', '.join(missing)}")
        if unknown:
            raise SystemExit(f"{sheet.name}: unknown names {unknown}. Valid names are in {NAMES.name}")
        guesses.append(rows)

    n = len(key)
    m1 = sum(guesses[0][i] == key[i] for i in key)
    m2 = sum(guesses[1][i] == key[i] for i in key)
    agree = sum(guesses[0][i] == guesses[1][i] for i in key)
    agree_right = sum(guesses[0][i] == guesses[1][i] == key[i] for i in key)

    kept = [r for r in _read(HUMAN_EVAL) if r["task"] != TASK] if HUMAN_EVAL.is_file() else []
    ours = [
        {"task": TASK, "item": i, "reference": key[i],
         "member1": guesses[0][i], "member2": guesses[1][i]}
        for i in sorted(key, key=int)
    ]
    _write(HUMAN_EVAL, HUMAN_EVAL_FIELDS, ours + kept)

    lines = [
        f"member1_correct: {m1}/{n} ({m1 / n:.1%})",
        f"member2_correct: {m2}/{n} ({m2 / n:.1%})",
        f"both_agreed: {agree}/{n} ({agree / n:.1%})",
        f"both_agreed_and_correct: {agree_right}/{n}",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", HUMAN_EVAL, "and", SUMMARY)


def _read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    commands = {"generate": generate, "score": score}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: python part_c_diffusion/blind_test.py generate|score")
    commands[sys.argv[1]]()
