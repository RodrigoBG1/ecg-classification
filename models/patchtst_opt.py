from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class CNNStem(nn.Module):
    """Replace linear patch projection with a shallow 1-D CNN.

    Processes each patch independently (groups=1) via three conv layers so
    local R-peak / P-wave / T-wave morphology is captured before attention.

    Input : (B_leads, N_patches, patch_size)
    Output: (B_leads, N_patches, d_model)
    """

    def __init__(self, patch_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        mid = max(d_model // 2, 32)
        # We treat each patch as a 1-D signal of length patch_size
        self.net = nn.Sequential(
            # Conv over raw patch samples
            nn.Conv1d(1, mid, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (*, mid, 1)
        )
        self.proj = nn.Linear(mid, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B_leads, N, patch_size)
        BL, N, P = x.shape
        x = x.reshape(BL * N, 1, P)          # treat patch as 1-D channel-1 signal
        x = self.net(x).squeeze(-1)           # (BL*N, mid)
        x = self.proj(x)                      # (BL*N, d_model)
        x = x.reshape(BL, N, -1)             # (BL, N, d_model)
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding (linear fallback — simpler, lighter)
# ─────────────────────────────────────────────────────────────────────────────

class LinearPatchEmbedding(nn.Module):
    """Simple linear projection of raw patch samples → d_model."""

    def __init__(self, patch_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B_leads, N, patch_size)
        return self.dropout(self.proj(x))     # (B_leads, N, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Learnable positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_patches: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointed encoder layer wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _CheckpointedLayer(nn.Module):
    """Wraps a TransformerEncoderLayer so grad-checkpoint can be applied."""

    def __init__(self, layer: nn.TransformerEncoderLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        # gradient checkpointing: recompute activations on backward
        # use_reentrant=False avoids deprecation warning in PyTorch ≥ 2.0
        return cp.checkpoint(self.layer, x, use_reentrant=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """Memory-efficient PatchTST for multi-label ECG classification.

    Input : (B, n_leads, L)      L = 5000 at 500 Hz, 1000 at 100 Hz
    Output: (B, n_classes)       raw logits (no sigmoid)

    Args
    ────
    n_classes       : number of output labels (5 for PTB-XL+ superclasses)
    n_leads         : ECG leads (12)
    signal_length   : samples per lead (5000 @ 500 Hz)
    patch_size      : samples per patch — 100 recommended @ 500 Hz (200 ms)
    stride          : patch hop — equal to patch_size = no overlap (lighter)
    d_model         : transformer hidden dim (128 recommended for CPU)
    n_heads         : attention heads (must divide d_model)
    n_layers        : transformer depth (4 recommended)
    ff_dim          : feed-forward hidden dim (256–512)
    attn_dropout    : dropout inside attention
    dropout         : dropout on embeddings and head
    use_cnn_stem    : replace linear proj with shallow CNN (recommended, +AUC)
    use_checkpoint  : gradient checkpointing per layer (saves RAM, default True)
    lead_chunk      : process this many leads per forward micro-batch (default 4)
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_leads: int = 12,
        signal_length: int = 5000,
        patch_size: int = 100,      # ← 100 instead of 50: N_patches = 50
        stride: int = 100,
        d_model: int = 128,         # ← 128 instead of 256
        n_heads: int = 4,           # ← 4 instead of 8
        n_layers: int = 4,          # ← 4 instead of 6
        ff_dim: int = 256,          # ← 256 instead of 1024
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        use_cnn_stem: bool = True,  # ← CNN stem for better AUC
        use_checkpoint: bool = True,
        lead_chunk: int = 4,        # ← process 4 leads at a time
    ) -> None:
        super().__init__()

        self.n_leads = n_leads
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.lead_chunk = lead_chunk

        self.n_patches = math.floor((signal_length - patch_size) / stride) + 1

        # ── Patch embedding ──────────────────────────────────────────────
        if use_cnn_stem:
            self.patch_embed: nn.Module = CNNStem(patch_size, d_model, dropout)
        else:
            self.patch_embed = LinearPatchEmbedding(patch_size, d_model, dropout)

        self.pos_enc = LearnablePositionalEncoding(self.n_patches, d_model)

        # ── Transformer encoder ──────────────────────────────────────────
        # Build layers; optionally wrap with gradient checkpointing
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            layers.append(_CheckpointedLayer(layer) if use_checkpoint else layer)

        self.encoder = nn.Sequential(*layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────
    def _encode_chunk(self, x_chunk: Tensor) -> Tensor:
        """Encode a chunk of leads.

        Args:
            x_chunk: (B_leads, L)  — already flattened batch × leads
        Returns:
            (B_leads, d_model)
        """
        patches = x_chunk.unfold(-1, self.patch_size, self.stride)  # (BL, N, P)
        x = self.patch_embed(patches)           # (BL, N, d_model)
        x = self.pos_enc(x)                     # (BL, N, d_model)
        x = self.encoder(x)                     # (BL, N, d_model)
        x = self.encoder_norm(x)
        return x.mean(dim=1)                    # (BL, d_model)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n_leads, L)
        Returns:
            logits: (B, n_classes)
        """
        B, C, L = x.shape
        lead_reps: list[Tensor] = []

        # Process leads in small chunks to cap peak memory
        for start in range(0, C, self.lead_chunk):
            chunk = x[:, start:start + self.lead_chunk, :]   # (B, chunk, L)
            chunk_size = chunk.shape[1]
            chunk = chunk.reshape(B * chunk_size, L)          # (B*chunk, L)
            rep = self._encode_chunk(chunk)                   # (B*chunk, d_model)
            rep = rep.reshape(B, chunk_size, self.d_model)    # (B, chunk, d_model)
            lead_reps.append(rep)

        # Pool over all leads
        lead_reps_cat = torch.cat(lead_reps, dim=1)           # (B, C, d_model)
        pooled = lead_reps_cat.mean(dim=1)                    # (B, d_model)

        return self.head(pooled)                              # (B, n_classes)

    # ─────────────────────────────────────────────────────────────────────
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ─────────────────────────────────────────────────────────────────────
    # Attention extraction — for interpretability / paper figures
    # ─────────────────────────────────────────────────────────────────────

    def get_attention_maps(
        self,
        x: Tensor,
        layer_idx: int = -1,
        lead_idx: int = 0,
    ) -> Tensor:
        """Extract multi-head attention weights from one encoder layer.

        Registers a temporary forward hook on the requested
        TransformerEncoderLayer's self-attention module, runs a single
        forward pass, then removes the hook.  Gradient checkpointing is
        automatically disabled for this call so the hook fires normally.

        Args:
            x         : (B, n_leads, L) — input batch (typically B=1 for viz).
            layer_idx : which encoder layer to tap (default -1 = last layer).
            lead_idx  : which ECG lead to extract attention from (0-11).

        Returns:
            attn_weights : (B, n_heads, N_patches, N_patches)
                           averaged over the batch dimension if B > 1.

        Usage example
        ─────────────
            model.eval()
            with torch.no_grad():
                attn = model.get_attention_maps(x.unsqueeze(0), layer_idx=-1, lead_idx=1)
            # attn shape: (1, n_heads, N, N)
        """
        # Resolve negative index
        n_layers = len(self.encoder)
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if not (0 <= layer_idx < n_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for encoder with {n_layers} layers."
            )

        # Unwrap _CheckpointedLayer if present to get the raw TransformerEncoderLayer
        raw_layer = self.encoder[layer_idx]
        if isinstance(raw_layer, _CheckpointedLayer):
            raw_layer = raw_layer.layer

        captured: list[Tensor] = []

        # We hook the TransformerEncoderLayer itself (not self_attn) so we
        # intercept the layer's *input* embeddings, then re-run only
        # self_attn with need_weights=True in a separate no-grad call.
        # The hook is registered BEFORE the layer runs, so self_attn has
        # not yet been called — no recursion.
        def _pre_hook(module, inp):
            # inp[0] is the src tensor (sequence embeddings)
            src = inp[0]
            with torch.no_grad():
                _, w = module.self_attn(
                    src, src, src,
                    need_weights=True,
                    average_attn_weights=False,
                )
            # w: (B_leads, n_heads, N, N)
            captured.append(w.detach().cpu())

        handle = raw_layer.register_forward_pre_hook(_pre_hook)

        self.eval()
        was_training = self.training
        try:
            with torch.no_grad():
                B, C, L = x.shape
                # Only process the requested lead (no need to chunk all 12)
                lead = x[:, lead_idx : lead_idx + 1, :]         # (B, 1, L)
                lead_flat = lead.reshape(B, L)                   # (B, L)
                patches = lead_flat.unfold(-1, self.patch_size, self.stride)  # (B, N, P)
                emb = self.patch_embed(patches)                  # (B, N, d_model)
                emb = self.pos_enc(emb)                          # (B, N, d_model)
                # Run through all layers up to and including the target layer
                # (need to bypass _CheckpointedLayer wrappers to fire the hook)
                h = emb
                for i, enc_layer in enumerate(self.encoder):
                    raw = enc_layer.layer if isinstance(enc_layer, _CheckpointedLayer) else enc_layer
                    h = raw(h)
                    if i == layer_idx:
                        break
        finally:
            handle.remove()
            if was_training:
                self.train()

        if not captured:
            raise RuntimeError("Attention hook did not fire. Check layer_idx.")

        # captured[0]: (B, n_heads, N, N)
        return captured[0]

    # ─────────────────────────────────────────────────────────────────────

    def plot_attention_maps(
        self,
        x: Tensor,
        layer_idx: int = -1,
        lead_idx: int = 0,
        sample_idx: int = 0,
        lead_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """Visualise per-head attention maps as a grid of heatmaps.

        Creates one subplot per attention head showing the N×N attention
        matrix for the requested sample and lead.  A mean-head panel is
        added at the end.

        Args:
            x          : (B, n_leads, L) input tensor.
            layer_idx  : encoder layer to visualise (default: last layer).
            lead_idx   : ECG lead index 0-11.
            sample_idx : which sample in the batch to plot (default 0).
            lead_names : optional list of 12 lead name strings for the title.
            save_path  : if given, saves the figure to this path (PNG/PDF).

        Returns:
            matplotlib Figure object (display with plt.show() or in a notebook).

        Raises:
            ImportError if matplotlib is not installed.

        Example (Jupyter / script)
        ──────────────────────────
            fig = model.plot_attention_maps(x_batch, layer_idx=-1, lead_idx=1)
            plt.show()
            # Or save directly:
            model.plot_attention_maps(x_batch, save_path="attn_lead_II.png")
        """
        if not _MPL_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plot_attention_maps(). "
                "Install it with:  pip install matplotlib"
            )

        _DEFAULT_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
        if lead_names is None:
            lead_names = _DEFAULT_LEAD_NAMES

        # ── Fetch attention weights ──────────────────────────────────────
        attn = self.get_attention_maps(x, layer_idx=layer_idx, lead_idx=lead_idx)
        # attn: (B, n_heads, N, N) — already on CPU
        attn_sample = attn[sample_idx]           # (n_heads, N, N)
        n_heads = attn_sample.shape[0]

        # ── Layout: n_heads + 1 mean panel ──────────────────────────────
        n_cols = min(n_heads + 1, 5)
        n_rows = math.ceil((n_heads + 1) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.2 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        patch_labels = [f"p{i}" for i in range(self.n_patches)]
        cmap = "viridis"

        for h in range(n_heads):
            ax = axes_flat[h]
            im = ax.imshow(
                attn_sample[h].numpy(), vmin=0, vmax=1,
                cmap=cmap, aspect="auto", interpolation="nearest",
            )
            ax.set_title(f"Head {h + 1}", fontsize=9, pad=3)
            ax.set_xticks(range(0, self.n_patches, max(1, self.n_patches // 5)))
            ax.set_yticks(range(0, self.n_patches, max(1, self.n_patches // 5)))
            ax.tick_params(labelsize=7)
            if h % n_cols == 0:
                ax.set_ylabel("Query patch", fontsize=7)
            ax.set_xlabel("Key patch", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mean-head panel
        ax_mean = axes_flat[n_heads]
        mean_attn = attn_sample.mean(dim=0).numpy()   # (N, N)
        im_mean = ax_mean.imshow(
            mean_attn, vmin=0, vmax=mean_attn.max(),
            cmap=cmap, aspect="auto", interpolation="nearest",
        )
        ax_mean.set_title("Mean (all heads)", fontsize=9, pad=3)
        ax_mean.set_xticks(range(0, self.n_patches, max(1, self.n_patches // 5)))
        ax_mean.set_yticks(range(0, self.n_patches, max(1, self.n_patches // 5)))
        ax_mean.tick_params(labelsize=7)
        ax_mean.set_xlabel("Key patch", fontsize=7)
        fig.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)

        # Hide any leftover empty subplots
        for ax in axes_flat[n_heads + 1:]:
            ax.set_visible(False)

        n_layers = len(self.encoder)
        resolved_layer = (n_layers + layer_idx) if layer_idx < 0 else layer_idx
        lead_label = lead_names[lead_idx] if lead_idx < len(lead_names) else str(lead_idx)
        fig.suptitle(
            f"PatchTST attention maps — lead {lead_label}  "
            f"(encoder layer {resolved_layer + 1}/{n_layers})",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved attention figure → {save_path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────

    def get_cls_attention(
        self,
        x: Tensor,
        layer_idx: int = -1,
        lead_idx: int = 0,
    ) -> Tensor:
        """Return mean-head attention vector over patches (N_patches,).

        Averages the attention matrix over heads and over query positions,
        producing a single importance score per patch.  Useful for plotting
        a 1-D saliency bar over the ECG time axis.

        Args:
            x         : (B, n_leads, L)
            layer_idx : encoder layer (default last).
            lead_idx  : ECG lead index.

        Returns:
            patch_importance : (N_patches,) — values in [0, 1], sum = 1.

        Example
        ───────
            importance = model.get_cls_attention(x.unsqueeze(0), lead_idx=1)
            # Map back to time: patch i covers samples [i*stride, i*stride+patch_size]
            plt.bar(range(len(importance)), importance.numpy())
        """
        attn = self.get_attention_maps(x, layer_idx=layer_idx, lead_idx=lead_idx)
        # attn: (B, n_heads, N, N)
        # Average over batch, heads, and query positions → (N,)
        importance = attn.mean(dim=0).mean(dim=0).mean(dim=0)
        importance = importance / importance.sum()
        return importance


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke tests …\n")

    configs = [
        # (label,             sr,  L,    patch, d_model, layers, heads, ff)
        ("light (CPU-safe)",  500, 5000, 100,   128,     4,      4,     256),
        ("medium",            500, 5000, 100,   192,     4,      4,     512),
        ("100 Hz light",      100, 1000,  10,   128,     4,      4,     256),
    ]

    for label, sr, L, ps, dm, nl, nh, ff in configs:
        model = PatchTST(
            n_classes=5,
            signal_length=L,
            patch_size=ps,
            stride=ps,
            d_model=dm,
            n_heads=nh,
            n_layers=nl,
            ff_dim=ff,
            use_cnn_stem=True,
            use_checkpoint=True,
            lead_chunk=4,
        )
        x = torch.randn(4, 12, L)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 5), f"Bad shape: {logits.shape}"

        # Estimate peak attention tensor size (single layer, no checkpointing)
        # (B*lead_chunk) × N² × 4 bytes
        bl = 4 * 4
        n = model.n_patches
        attn_mb = bl * n * n * 4 / 1e6
        print(
            f"[{label:20s}]  sr={sr}  patch={ps}  N={n}  "
            f"d={dm}  layers={nl}  params={model.n_parameters:>8,}  "
            f"attn_peak≈{attn_mb:.1f} MB/layer"
        )

    print("\nAll forward-pass smoke tests passed.")

    # ── Attention extraction smoke test ──────────────────────────────────
    print("\nTesting attention extraction …")
    model_attn = PatchTST(
        n_classes=5, signal_length=5000, patch_size=100, stride=100,
        d_model=128, n_heads=4, n_layers=4, ff_dim=256,
        use_cnn_stem=True,
        use_checkpoint=False,   # hooks don't fire through gradient checkpoint
        lead_chunk=4,
    )
    model_attn.eval()
    x_test = torch.randn(2, 12, 5000)

    with torch.no_grad():
        # get_attention_maps: last encoder layer, lead I (index 0)
        attn = model_attn.get_attention_maps(x_test, layer_idx=-1, lead_idx=0)
        assert attn.shape == (2, 4, 50, 50), f"Unexpected shape: {attn.shape}"
        print(f"  get_attention_maps → {attn.shape}  ✓")

        # get_cls_attention: patch importance vector
        imp = model_attn.get_cls_attention(x_test, layer_idx=-1, lead_idx=1)
        assert imp.shape == (50,), f"Unexpected shape: {imp.shape}"
        assert abs(imp.sum().item() - 1.0) < 1e-5, "Importance does not sum to 1"
        print(f"  get_cls_attention  → {imp.shape}  sum={imp.sum():.4f}  ✓")

        # plot_attention_maps (skipped if matplotlib not available)
        if _MPL_AVAILABLE:
            fig = model_attn.plot_attention_maps(
                x_test, layer_idx=-1, lead_idx=0,
                save_path="/tmp/attn_smoke_test.png",
            )
            print(f"  plot_attention_maps → saved to /tmp/attn_smoke_test.png  ✓")
            plt.close(fig)
        else:
            print("  plot_attention_maps skipped (matplotlib not installed)")

    print("\nAll smoke tests passed.")
