"""Diversity metrics for the Demiurge scene dataset.

Computes mean pairwise distance (MPD) in two embedding spaces:

1. Hand-crafted scene encoding: per-slot concatenation of object-type
   one-hot (N_CLASSES + 1 dims), XYZ position (3 dims), and presence
   bit (1 dim). Padded/absent slots contribute zero vectors.
   Total feature dim: N_MAX * (N_CLASSES + 1 + 3 + 1) = 12 * 17 = 204.

2. Frozen sentence-transformer text embeddings: encodes task descriptions
   with ``all-MiniLM-L6-v2`` (384-dim, L2-normalised by the model).

MPD is the mean of all pairwise L2 distances in the sample. For N samples,
computed as mean(upper triangle of the N x N distance matrix) via the
identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>.

Both metrics are the baselines that Week 3 learned generation must beat.

Usage::

    from data.reader import ShardReader
    from data.diversity import compute_diversity

    result = compute_diversity(ShardReader("data/v1/"), n_sample=5000, seed=0)
    print(result.scene_encoding_mpd, result.text_embedding_mpd)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from data.reader import ShardReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MAX: int = 12          # Fixed object slots per scene (matches schema.py)
N_CLASSES: int = 12      # Vocab size -- IDs 0-11 (matches vocab.py)
# Slot encoding: one-hot(N_CLASSES + 1 PAD) + xyz + presence
_SLOT_DIM: int = N_CLASSES + 1 + 3 + 1   # = 17
SCENE_DIM: int = N_MAX * _SLOT_DIM        # = 204


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiversityResult:
    """Mean pairwise distances in two embedding spaces.

    Attributes:
        scene_encoding_mpd: Mean pairwise L2 distance in hand-crafted
            scene encoding space (dim=204). Range: nominally (0, 100).
        text_embedding_mpd: Mean pairwise L2 distance in
            all-MiniLM-L6-v2 embedding space (dim=384). Range: (0, 2.0]
            because the model returns L2-normalised vectors.
        n_sampled: Number of scenes actually used (may be < n_sample if
            the reader exhausts before reaching n_sample).
        seed: Random seed used for reservoir sampling.
    """

    scene_encoding_mpd: float
    text_embedding_mpd: float
    n_sampled: int
    seed: int


# ---------------------------------------------------------------------------
# Scene encoding
# ---------------------------------------------------------------------------


def encode_scene(scene: object) -> np.ndarray:
    """Encode a SceneTensor into a fixed-length numpy vector.

    Args:
        scene: A ``SceneTensor`` instance with ``object_types`` (LongTensor
            [N_MAX]), ``poses`` (FloatTensor [N_MAX, 7]), and ``presence``
            (BoolTensor [N_MAX]) attributes.

    Returns:
        numpy array of shape (SCENE_DIM,) = (204,).
    """
    types: torch.Tensor = scene.object_types.long()   # [N_MAX]
    poses: torch.Tensor = scene.poses.float()          # [N_MAX, 7]
    presence: torch.Tensor = scene.presence.bool()     # [N_MAX]

    n = types.shape[0]
    assert n == N_MAX, f"Expected N_MAX={N_MAX} slots, got {n}"

    slots: list[np.ndarray] = []
    for i in range(N_MAX):
        present = bool(presence[i].item())
        if present:
            # One-hot over N_CLASSES + 1 (index N_CLASSES reserved for PAD).
            one_hot = np.zeros(N_CLASSES + 1, dtype=np.float32)
            type_id = int(types[i].item())
            one_hot[min(type_id, N_CLASSES)] = 1.0
            xyz = poses[i, :3].numpy().astype(np.float32)
            pres = np.array([1.0], dtype=np.float32)
        else:
            one_hot = np.zeros(N_CLASSES + 1, dtype=np.float32)
            one_hot[N_CLASSES] = 1.0   # PAD class
            xyz = np.zeros(3, dtype=np.float32)
            pres = np.zeros(1, dtype=np.float32)

        slots.append(np.concatenate([one_hot, xyz, pres]))

    return np.concatenate(slots)  # (SCENE_DIM,)


# ---------------------------------------------------------------------------
# Pairwise distance computation
# ---------------------------------------------------------------------------


def _mean_pairwise_l2(matrix: np.ndarray) -> float:
    """Compute mean pairwise L2 distance for a row-matrix of vectors.

    Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b> to avoid
    forming an explicit NxN difference tensor.

    Args:
        matrix: shape (N, D).

    Returns:
        Mean of the upper triangle (excluding diagonal) of the N x N
        distance matrix. Returns 0.0 if N < 2.
    """
    n = matrix.shape[0]
    if n < 2:
        return 0.0

    # Squared norms: (N,)
    sq_norms = (matrix ** 2).sum(axis=1)
    # Gram matrix: (N, N)
    gram = matrix @ matrix.T
    # Squared distance matrix: (N, N)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    # Clamp negatives caused by floating-point error before sqrt.
    np.clip(sq_dist, 0.0, None, out=sq_dist)
    dist = np.sqrt(sq_dist)

    # Mean of upper triangle (N*(N-1)/2 pairs).
    upper = dist[np.triu_indices(n, k=1)]
    return float(upper.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_diversity(
    reader: ShardReader,
    n_sample: int = 5000,
    seed: int = 0,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> DiversityResult:
    """Compute mean pairwise distance in scene encoding and text embedding spaces.

    Samples up to ``n_sample`` scenes from ``reader`` using reservoir
    sampling with a fixed seed for reproducibility. The reader is consumed
    once; if it has fewer than ``n_sample`` examples the actual sample size
    will be smaller.

    Args:
        reader: An open ``ShardReader`` instance over the dataset.
        n_sample: Target sample size. Defaults to 5000.
        seed: Random seed for reservoir sampling and reproducibility.
        text_model_name: HuggingFace model name for the sentence-transformer.
        device: PyTorch device string for text encoding.

    Returns:
        A :class:`DiversityResult` with both MPD values.
    """
    rng = random.Random(seed)

    # Reservoir sampling -- keep a uniform random sample of n_sample items
    # without loading the full dataset into memory.
    reservoir_scenes: list[object] = []
    reservoir_descs: list[str] = []

    for i, (scene, desc, _report, _sdf) in enumerate(reader):
        if i < n_sample:
            reservoir_scenes.append(scene)
            reservoir_descs.append(desc)
        else:
            j = rng.randint(0, i)
            if j < n_sample:
                reservoir_scenes[j] = scene
                reservoir_descs[j] = desc

    n_actual = len(reservoir_scenes)

    # --- Scene encoding MPD ---
    scene_matrix = np.stack(
        [encode_scene(s) for s in reservoir_scenes], axis=0
    )  # (n_actual, SCENE_DIM)
    scene_mpd = _mean_pairwise_l2(scene_matrix)

    # --- Text embedding MPD ---
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(text_model_name, device=device)
    model.eval()

    with torch.no_grad():
        text_embeddings: np.ndarray = model.encode(
            reservoir_descs,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    # text_embeddings: (n_actual, 384) -- L2-normalised by SentenceTransformer

    text_mpd = _mean_pairwise_l2(text_embeddings)

    return DiversityResult(
        scene_encoding_mpd=scene_mpd,
        text_embedding_mpd=text_mpd,
        n_sampled=n_actual,
        seed=seed,
    )
