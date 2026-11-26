"""§3.5 Human review of 20 RAG answers (fixed seed), by both members.

After eval_generation.py, from the repo root (no GPU needed):

    python part_d_rag/human_eval.py generate

This picks 20 of the 100 answers and writes:
  results/rag_review.md              each question, its 5 passages, the answer, the reference
  results/rag_review_member1.csv     sheet for member 1 (qid, supported, correct)
  results/rag_review_member2.csv     sheet for member 2

Each member fills in their own sheet alone with yes/no:
  supported = everything the answer says is in the passages
  correct   = the answer agrees with the reference answer
Then:

    python part_d_rag/human_eval.py score

This writes the rag_supported and rag_correct rows of results/human_eval.csv
(other rows are kept) and the Table 5 numbers to results/part_d_human_summary.txt.
"""
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
ANSWERS = RESULTS / "rag_answers.jsonl"
REVIEW = RESULTS / "rag_review.md"
SHEETS = [RESULTS / "rag_review_member1.csv", RESULTS / "rag_review_member2.csv"]
HUMAN_EVAL = RESULTS / "human_eval.csv"
SUMMARY = RESULTS / "part_d_human_summary.txt"
HUMAN_EVAL_FIELDS = ["task", "item", "reference", "member1", "member2"]
TASKS = ("rag_supported", "rag_correct")
N_REVIEW = 20
SEED = 0


def generate():
    records = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()]
    for sheet in SHEETS:
        if sheet.is_file() and any(r["supported"] or r["correct"] for r in _read(sheet)):
            raise SystemExit(f"{sheet} already has marks. Delete it to start a new review.")
    picked = sorted(random.Random(SEED).sample(records, N_REVIEW), key=lambda r: r["qid"])

    blocks = ["# Part D: human review of 20 answers\n",
              "Mark each answer in your own sheet, without looking at the other member's.\n",
              "- **supported**: everything the answer says is in the passages",
              "- **correct**: the answer agrees with the reference answer\n"]
    for r in picked:
        blocks.append(f"## Q{r['qid']}: {r['question']}\n")
        for p in r["passages"]:
            blocks.append(f"[{p['n']}] (abstract {p['abstract_id']}) {p['text']}\n")
        blocks.append(f"**Answer:** {r['answer']}\n")
        blocks.append(f"**Reference answer:** {r['reference']}\n")
    REVIEW.write_text("\n".join(blocks), encoding="utf-8")
    for sheet in SHEETS:
        _write(sheet, ["qid", "supported", "correct"],
               [{"qid": r["qid"], "supported": "", "correct": ""} for r in picked])
    print("saved", REVIEW, "and", ", ".join(s.name for s in SHEETS))


def score():
    marks = []
    for sheet in SHEETS:
        rows = _read(sheet)
        for r in rows:
            for col in ("supported", "correct"):
                r[col] = r[col].strip().lower()
                if r[col] not in ("yes", "no"):
                    raise SystemExit(f"{sheet.name}: qid {r['qid']} {col} must be yes or no")
        marks.append({r["qid"]: r for r in rows})
    if marks[0].keys() != marks[1].keys():
        raise SystemExit("The two sheets list different questions.")
    qids = sorted(marks[0], key=int)

    kept = [r for r in _read(HUMAN_EVAL) if r["task"] not in TASKS] if HUMAN_EVAL.is_file() else []
    ours = [
        {"task": f"rag_{col}", "item": qid, "reference": "",
         "member1": marks[0][qid][col], "member2": marks[1][qid][col]}
        for col in ("supported", "correct") for qid in qids
    ]
    _write(HUMAN_EVAL, HUMAN_EVAL_FIELDS, kept + ours)

    n = len(qids)
    lines = []
    for col in ("supported", "correct"):
        counts = [sum(m[q][col] == "yes" for q in qids) for m in marks]
        lines.append(f"{col}: member1 {counts[0]}/{n}, member2 {counts[1]}/{n}")
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
        raise SystemExit("usage: python part_d_rag/human_eval.py generate|score")
    commands[sys.argv[1]]()
