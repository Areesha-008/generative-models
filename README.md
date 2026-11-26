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

## Train Parts A–C

On Colab (A100), open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) and run it top to bottom. It mounts Drive, clones the repo, downloads UHaT, runs the starter check and the numerical checks, and trains all three models. Checkpoints go to Drive. `results/` is copied to Drive after each part.

To run the same steps by hand, from the repo root:

```bash
python part_a_mae/model.py            # self-checks
python part_b_vae/model.py
python part_c_diffusion/diffusion.py
python part_a_mae/pretrain.py         # 100 epochs, then reconstructions
python part_a_mae/classify.py         # few-label classifier, accuracy, confusion matrix
python part_b_vae/train.py            # 30 epochs, reconstructions, samples, interpolation
python part_c_diffusion/train.py      # 40 epochs, forward noising, 40x8 samples, denoising
python part_c_diffusion/blind_test.py generate   # one shuffled sample per character
```

**Blind test (§3.3).** `generate` writes `results/part_c_blind_test.png`, which has 40 numbered images, and one blank sheet per member: `results/blind_member1.csv` and `results/blind_member2.csv`. Each member fills in the `guess` column alone, using names from `results/part_c_blind_test_names.txt`. Don't open `results/part_c_blind_key.csv` until both sheets are done. Then run:

```bash
python part_c_diffusion/blind_test.py score
```

This writes the `diffusion_blind` rows of `results/human_eval.csv` and the Table 3 numbers to `results/part_c_blind_test_summary.txt`.

On a GPU the training data is held in GPU memory, and Parts A and C train under bfloat16 autocast. Set `AMP=0` to train in float32. Each script saves its checkpoint after every epoch to `part_x/checkpoints/`, which git ignores. It also writes parameters, final loss and time to `results/part_x_training.txt` for Table 2.

## Numerical checks

These catch the usual bugs. Each implementation has to satisfy them.

- MAE masks exactly 48 of 64 patches, and the loss ignores the visible ones.
- The VAE's KL is never negative, and it is 0 when μ = 0 and log σ² = 0.
- At the last diffusion step, ᾱ_T is about 4×10⁻⁵, so the image is almost pure noise.
- Every RAG chunk is at most 150 words, every embedding has length 1, and a perfect ranking scores 1 on Hit, Recall, and MRR.
