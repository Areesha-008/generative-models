"""Inference for the four app tabs. No UI code here, so it can be tested alone.

Every model is loaded the first time its tab is used, then kept in memory.
Weights come from the paths below (download them from the links in the README,
or train them with the scripts in each part folder).
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the part folders and starter/ import as packages

from part_a_mae.model import LinearClassifier, MaskedAutoencoder  # noqa: E402
from part_b_vae.model import LATENT, VAE  # noqa: E402
from part_c_diffusion.diffusion import Schedule, make_unet, sample, to_unit  # noqa: E402
from part_d_rag.encoder import Encoder  # noqa: E402
from part_d_rag.generator import MEDICAL_NOTICE, TOP_CHUNKS, Generator  # noqa: E402
from part_d_rag.index import INDEX_DIR, load_index  # noqa: E402
from part_d_rag.metrics import cited_numbers, is_idk  # noqa: E402
from part_d_rag.search import Retriever  # noqa: E402
from starter.eurosat import CLASS_NAMES, load_eurosat_split, patchify, unpatchify  # noqa: E402
from starter.uhat import _find_split_dir, load_uhat_characters  # noqa: E402

MAE_CKPT = ROOT / "part_a_mae" / "checkpoints" / "mae.pt"
HEAD_CKPT = ROOT / "part_a_mae" / "checkpoints" / "classifier_head.pt"
VAE_CKPT = ROOT / "part_b_vae" / "checkpoints" / "vae.pt"
UNET_CKPT = ROOT / "part_c_diffusion" / "checkpoints" / "unet.pt"
UHAT_DIR = ROOT / "starter" / "data" / "uhat"

DISPLAY = 256  # pixels; small images are scaled up with nearest-neighbour


def pick_device():
    """APP_DEVICE overrides; otherwise cuda, then Apple mps, then cpu."""
    if os.environ.get("APP_DEVICE"):
        return torch.device(os.environ["APP_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


class MissingFile(Exception):
    """A checkpoint, dataset or index the app needs is not on disk."""


def _load_state(model, path):
    if not path.is_file():
        raise MissingFile(f"Missing {path.relative_to(ROOT)}. See 'Run the app' in the README.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def to_pil(image, size=DISPLAY):
    """(C, H, W) tensor in [0, 1] -> PIL image scaled to `size` pixels wide."""
    array = (image.detach().float().cpu().clamp(0, 1) * 255).round().byte()
    array = array.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    pil = Image.fromarray(array)
    return pil.resize((size, size * pil.height // pil.width), Image.NEAREST)


def grid(images, nrow, pad=2):
    """(N, C, H, W) in [0, 1] -> one (C, H', W') tile grid with a grey border."""
    n, c, h, w = images.shape
    rows = (n + nrow - 1) // nrow
    out = torch.full((c, rows * (h + pad) + pad, nrow * (w + pad) + pad), 0.5)
    for i in range(n):
        r, col = divmod(i, nrow)
        top, left = pad + r * (h + pad), pad + col * (w + pad)
        out[:, top:top + h, left:left + w] = images[i].cpu()
    return out


# ---------------------------------------------------------------- Part A: MAE

@lru_cache(maxsize=1)
def mae_models():
    mae = _load_state(MaskedAutoencoder(), MAE_CKPT)
    clf = LinearClassifier(mae)
    _load_state(clf.head, HEAD_CKPT)
    return mae, clf.to(DEVICE).eval()


@lru_cache(maxsize=1)
def eurosat_test():
    """The 5.4k EuroSAT test tiles (downloaded by `datasets` the first time)."""
    return load_eurosat_split("test")


@torch.no_grad()
def mae_demo(index, mask_seed):
    """Masked input, reconstruction, original, class probabilities, true class."""
    images, labels = eurosat_test()
    index = int(index) % len(images)
    mae, clf = mae_models()
    image = images[index:index + 1]
    patches = patchify(image).to(DEVICE)

    torch.manual_seed(int(mask_seed))
    pred, mask = mae(patches)
    hidden = mask.bool().unsqueeze(-1)
    masked = unpatchify(torch.where(hidden, torch.full_like(patches, 0.5), patches))
    recon = unpatchify(torch.where(hidden, pred, patches).clamp(0, 1))

    # The classifier sees the whole tile, as in A.3–A.4.
    probs = torch.softmax(clf(patches).float(), dim=1)[0].cpu()
    return (
        to_pil(masked[0]),
        to_pil(recon[0]),
        to_pil(image[0]),
        {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        CLASS_NAMES[int(labels[index])],
    )


# ---------------------------------------------------------------- Part B: VAE

@lru_cache(maxsize=1)
def uhat_names():
    """The 40 character names, in the order the models were trained with."""
    if not UHAT_DIR.is_dir():
        raise MissingFile(f"Missing {UHAT_DIR.relative_to(ROOT)}. Unzip UHaT there (README).")
    try:
        split_dir = _find_split_dir(UHAT_DIR, "char", "train")
    except FileNotFoundError as err:
        raise MissingFile(str(err)) from err
    # Same rule as starter/uhat.py, without loading 28k training images.
    return sorted(c.name.lower() for c in split_dir.iterdir() if c.is_dir())


@lru_cache(maxsize=1)
def uhat_test():
    images, labels, _ = load_uhat_characters(UHAT_DIR, split="test", class_names=uhat_names())
    return images, labels


@lru_cache(maxsize=1)
def vae_model():
    return _load_state(VAE(), VAE_CKPT)


@torch.no_grad()
def vae_random(seed, count=16):
    """`count` new characters decoded from z ~ N(0, I), as one 4-wide grid."""
    generator = torch.Generator().manual_seed(int(seed))
    z = torch.randn(count, LATENT, generator=generator).to(DEVICE)
    images = torch.sigmoid(vae_model().decoder(z))
    return to_pil(grid(images, nrow=4), size=DISPLAY * 2)


def _character(name, example):
    """One test image of character `name`. `example` picks which one."""
    images, labels = uhat_test()
    members = (labels == uhat_names().index(name)).nonzero(as_tuple=True)[0]
    return images[members[int(example) % len(members)]]


@torch.no_grad()
def vae_interpolate(name_a, name_b, weight, example=0, steps=10):
    """Decode (1 - w) * mu_a + w * mu_b. Returns A, the blend, B, and a 10-step strip."""
    pair = torch.stack([_character(name_a, example), _character(name_b, example)])
    vae = vae_model()
    mu, _ = vae.encoder(pair.to(DEVICE))
    blend = torch.sigmoid(vae.decoder(torch.lerp(mu[0], mu[1], float(weight)).unsqueeze(0)))
    w = torch.linspace(0, 1, steps, device=DEVICE).unsqueeze(1)
    strip = torch.sigmoid(vae.decoder((1 - w) * mu[0] + w * mu[1]))
    return (
        to_pil(pair[0]),
        to_pil(blend[0]),
        to_pil(pair[1]),
        to_pil(grid(strip, nrow=steps), size=DISPLAY * 3),
    )


# ---------------------------------------------------------- Part C: diffusion

@lru_cache(maxsize=1)
def unet_and_schedule():
    unet = _load_state(make_unet(num_classes=len(uhat_names())), UNET_CKPT)
    return unet, Schedule().to(DEVICE)


def diffusion_generate(name, seed, count=8):
    """`count` new samples of character `name`, 1000 DDPM steps each, as one grid."""
    unet, schedule = unet_and_schedule()
    labels = torch.full((count,), uhat_names().index(name), device=DEVICE, dtype=torch.long)
    torch.manual_seed(int(seed))
    x, _ = sample(unet, labels, schedule)
    return to_pil(grid(to_unit(x), nrow=4), size=DISPLAY * 2)


# ----------------------------------------------------------------- Part D: RAG

@lru_cache(maxsize=1)
def rag_parts():
    if not (INDEX_DIR / "embeddings.pt").is_file():
        raise MissingFile(f"Missing {INDEX_DIR.relative_to(ROOT)}. Run part_d_rag/index.py (README).")
    embeddings, texts, ids = load_index()
    retriever = Retriever(embeddings, texts, ids, device=DEVICE)
    return Encoder(device=DEVICE), retriever, Generator(device=DEVICE)


def rag_answer(question):
    """Returns the answer, the notice, and the 5 passages with a 'cited' flag."""
    question = question.strip()
    encoder, retriever, generator = rag_parts()
    query = encoder.encode([question])[0]
    passages = retriever.top_chunks(query, k=TOP_CHUNKS)
    answer = generator.answer([question], [passages])[0]
    cited = set(cited_numbers(answer))
    for n, p in enumerate(passages, start=1):
        p["n"] = n
        p["cited"] = n in cited
    return {
        "answer": answer,
        "notice": MEDICAL_NOTICE,
        "idk": is_idk(answer),
        "passages": passages,
    }

