# Generative models

Four separate research systems, trained apart and shown together in one web app.

| Part | Folder | Job |
| --- | --- | --- |
| A. Masked autoencoder | `part_a_mae/` | Reconstruct EuroSAT tiles, then classify land cover from 16 labels per class |
| B. Variational autoencoder | `part_b_vae/` | Compress a handwritten Urdu character into a 16-number code |
| C. Diffusion | `part_c_diffusion/` | Denoise toward a requested Urdu character |
| D. Retrieval-augmented generation | `part_d_rag/` | Retrieve PubMed abstracts and answer with citations |

The four models never share weights. They meet only in `app/`.

```
starter/            data loading and the starter check
part_a_mae/         MAE pretraining and few-label classifier
part_b_vae/         VAE
part_c_diffusion/   DDPM schedule, training, sampling
part_d_rag/         chunking, index, search, generation
results/            tables, figures, analysis.md
app/                four-tab front end
```

Weights are not committed. Links go here once the checkpoints exist.

## Setup

```bash
pip install -r requirements.txt
```

EuroSAT and BioASQ download from Hugging Face the first time they are loaded. For UHaT, download `data.zip` from Zenodo (or Kaggle) and unzip it into `starter/data/uhat/`. The loader finds the character train and test folders by name.

## Check the starter

```bash
cd starter
python check_starter.py
```

Expect:

- EuroSAT train `(16200, 3, 64, 64)`, test `(5400, 3, 64, 64)`
- few-label subset 160; patches for 8 images `(8, 64, 192)`
- UHaT train and test shapes, and 40 class names
- BioASQ about 40181 abstracts and 707 test questions

The two grids are written to `starter/results/`.

## Numerical checks

These catch the usual bugs. Each implementation has to satisfy them.

- MAE masks exactly 48 of 64 patches, and the loss ignores the visible ones.
- The VAE's KL is never negative, and it is 0 when μ = 0 and log σ² = 0.
- At the last diffusion step, ᾱ_T is about 4×10⁻⁵, so the image is almost pure noise.
- Every RAG chunk is at most 150 words, every embedding has length 1, and a perfect ranking scores 1 on Hit, Recall, and MRR.
