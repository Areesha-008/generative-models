"""D.1–D.2 Build the chunk index: chunk every abstract, embed every chunk, save to disk.

Run from the repo root (GPU recommended):

    python part_d_rag/index.py

Writes part_d_rag/index/embeddings.pt (float32, n_chunks x 384, unit rows) and
part_d_rag/index/chunks.json (chunk text and abstract id for each row).
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(HERE))

INDEX_DIR = HERE / "index"
EMB_PATH = INDEX_DIR / "embeddings.pt"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def save_index(embeddings, texts, abstract_ids, index_dir=INDEX_DIR):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, index_dir / EMB_PATH.name)
    with (index_dir / CHUNKS_PATH.name).open("w", encoding="utf-8") as f:
        json.dump({"texts": texts, "abstract_ids": abstract_ids}, f)


def load_index(index_dir=INDEX_DIR):
    """Returns (embeddings (C, 384) float32, chunk texts, chunk abstract ids)."""
    index_dir = Path(index_dir)
    if not (index_dir / EMB_PATH.name).is_file():
        raise SystemExit(f"No index in {index_dir}. Run part_d_rag/index.py first.")
    embeddings = torch.load(index_dir / EMB_PATH.name, weights_only=True)
    with (index_dir / CHUNKS_PATH.name).open(encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta["texts"], meta["abstract_ids"]


def main():
    from bioasq import load_corpus
    from chunking import chunk_corpus
    from encoder import Encoder

    corpus = load_corpus()
    texts, ids = chunk_corpus(corpus)
    print(f"abstracts: {len(corpus)}  chunks: {len(texts)}", flush=True)
    start = time.time()
    embeddings = Encoder().encode(texts, show_progress=True)
    print(f"embedded in {time.time() - start:.0f}s", flush=True)
    save_index(embeddings, texts, ids)
    norms = embeddings.norm(dim=1)
    print(f"saved {EMB_PATH} {tuple(embeddings.shape)}; norms {norms.min():.6f}..{norms.max():.6f}")


if __name__ == "__main__":
    main()
