"""Part A masked autoencoder, assembled from nn.TransformerEncoder.

Patch embed Linear(192, 192), plus a learned 192-d position vector per patch.
The encoder is 6 TransformerEncoderLayers (width 192, 3 heads, feed-forward
768) on the 16 visible patches, then LayerNorm.

The decoder projects those outputs with Linear(192, 128), fills every hidden
position with a learned mask token, adds a learned 128-d position vector, runs
2 TransformerEncoderLayers (width 128, 4 heads), and predicts pixels with
Linear(128, 192).

The brief sets the encoder feed-forward width at 768, which is 4 times 192.
It does not set the decoder feed-forward width, so that stack uses the same
ratio: 4 times 128 = 512. Dropout is 0. The brief does not ask for it, and the
PyTorch layer would otherwise drop features at random during training.
"""
import torch
from torch import nn

NUM_PATCHES = 64
PATCH_DIM = 192
DIM = 192
DECODER_DIM = 128
NUM_KEEP = 16
NUM_MASK = NUM_PATCHES - NUM_KEEP  # 48
ENCODER_DEPTH = 6
DECODER_DEPTH = 2
ENCODER_HEADS = 3
DECODER_HEADS = 4
ENCODER_FF = 768
DECODER_FF = DECODER_DIM * 4  # 512
NUM_CLASSES = 10


def _encoder(d_model, nhead, dim_feedforward, n_layers, final_norm):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        batch_first=True,
    )
    norm = nn.LayerNorm(d_model) if final_norm else None
    return nn.TransformerEncoder(
        layer, num_layers=n_layers, norm=norm, enable_nested_tensor=False
    )


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, DIM)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, DIM))
        self.encoder = _encoder(DIM, ENCODER_HEADS, ENCODER_FF, ENCODER_DEPTH, final_norm=True)
        self.to_decoder = nn.Linear(DIM, DECODER_DIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_DIM))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, DECODER_DIM))
        self.decoder = _encoder(
            DECODER_DIM, DECODER_HEADS, DECODER_FF, DECODER_DEPTH, final_norm=False
        )
        self.pixel_head = nn.Linear(DECODER_DIM, PATCH_DIM)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x):
        """Keep 16 patches. A new random choice for every image. mask is 1 on hidden patches."""
        n, length, _ = x.shape
        noise = torch.rand(n, length, device=x.device)
        ids_keep = torch.argsort(noise, dim=1)[:, :NUM_KEEP]
        mask = torch.ones(n, length, device=x.device, dtype=x.dtype)
        mask.scatter_(1, ids_keep, 0)
        return ids_keep, mask

    def _take(self, x, ids):
        idx = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, 1, idx)

    def encode_visible(self, patches, ids_keep):
        x = self.patch_embed(patches) + self.pos_embed
        return self.encoder(self._take(x, ids_keep))

    def encode_all(self, patches):
        """All 64 patches, no masking. (N, 64, 192), after the encoder LayerNorm."""
        x = self.patch_embed(patches) + self.pos_embed
        return self.encoder(x)

    def decode(self, encoded, ids_keep, mask):
        tokens = self.to_decoder(encoded)
        n, dim = tokens.shape[0], tokens.shape[-1]
        full = self.mask_token.expand(n, NUM_PATCHES, dim).clone()
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, dim)
        full = full.scatter(1, idx, tokens)
        full = full + self.decoder_pos_embed
        return self.pixel_head(self.decoder(full))

    def forward(self, patches):
        """Pixel predictions (N, 64, 192) and mask (N, 64), 1 = hidden."""
        ids_keep, mask = self.random_masking(patches)
        encoded = self.encode_visible(patches, ids_keep)
        return self.decode(encoded, ids_keep, mask), mask


def masked_mse(pred, target, mask):
    """Mean squared error per pixel value, over the hidden patches only."""
    per_patch = (pred - target).pow(2).mean(dim=-1)  # (N, 64)
    return (per_patch * mask).sum() / mask.sum()


class LinearClassifier(nn.Module):
    """Frozen encoder, mean over the 64 patch outputs, Linear(192, 10).

    Build this after pretraining. Only the linear layer is trained.
    """

    def __init__(self, mae, num_classes=NUM_CLASSES):
        super().__init__()
        self.mae = mae
        for p in self.mae.parameters():
            p.requires_grad = False
        self.mae.eval()
        self.head = nn.Linear(DIM, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.mae.eval()
        return self

    def forward(self, patches):
        with torch.no_grad():
            tokens = self.mae.encode_all(patches)
        return self.head(tokens.mean(dim=1))


if __name__ == "__main__":
    torch.manual_seed(0)
    mae = MaskedAutoencoder()
    patches = torch.rand(4, NUM_PATCHES, PATCH_DIM)
    pred, mask = mae(patches)
    assert pred.shape == (4, NUM_PATCHES, PATCH_DIM)
    assert torch.equal(mask.sum(dim=1), torch.full((4,), float(NUM_MASK)))
    assert len(mae.encoder.layers) == ENCODER_DEPTH
    assert len(mae.decoder.layers) == DECODER_DEPTH
    assert mae.encoder.layers[0].linear1.out_features == ENCODER_FF
    assert mae.decoder.layers[0].linear1.out_features == DECODER_FF
    assert mae.encoder.layers[0].self_attn.num_heads == ENCODER_HEADS
    assert mae.decoder.layers[0].self_attn.num_heads == DECODER_HEADS
    assert isinstance(mae.encoder.norm, nn.LayerNorm)
    assert mae.decoder.norm is None

    target = torch.rand_like(patches)
    loss = masked_mse(pred, target, mask)
    changed_visible = target.clone()
    changed_visible[mask == 0] = torch.rand_like(changed_visible[mask == 0])
    assert torch.allclose(loss, masked_mse(pred, changed_visible, mask))
    changed_hidden = target.clone()
    changed_hidden[mask == 1] = torch.rand_like(changed_hidden[mask == 1])
    assert not torch.allclose(loss, masked_mse(pred, changed_hidden, mask))
    # The loss is a true per-value mean: it equals plain MSE over the hidden patches.
    hidden = mask.bool()
    assert torch.allclose(loss, (pred[hidden] - target[hidden]).pow(2).mean())

    features = mae.encode_all(patches)
    assert features.shape == (4, NUM_PATCHES, DIM)
    clf = LinearClassifier(mae)
    clf.train()
    logits = clf(patches)
    assert logits.shape == (4, NUM_CLASSES)
    assert all(not p.requires_grad for p in clf.mae.parameters())
    assert clf.head.weight.requires_grad
    assert not clf.mae.training
    print("checks passed")
    print("hidden patches per image:", int(mask[0].sum()))
    print("encoder", len(mae.encoder.layers), "layers, decoder", len(mae.decoder.layers), "layers")
    print("classifier logits:", tuple(logits.shape))
