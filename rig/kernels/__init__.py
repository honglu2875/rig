"""Reusable JAX kernels shared by rig recipes.

The competition still keeps the model/training algorithm in one entry script;
these kernels are deliberately limited to fundamental, shape-generic building
blocks whose behavior can be validated independently.
"""

from .linear_cross_entropy import (
    DEFAULT_VOCAB_TILE_SIZE,
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)
from .fuzzy_topk import (
    FuzzyTopKCallable,
    FuzzyTopKConfig,
    fuzzy_topk_mlp,
    fuzzy_topk_relu,
    make_mesh_fuzzy_topk_mlp,
    naive_fuzzy_topk_mlp,
)
from .sparse_mlp import (
    SparseMlpCallable,
    SparseMlpConfig,
    make_mesh_sparse_topk_mlp,
    naive_dense_topk_mlp,
    pallas_sparse_decode,
    reference_sparse_decode,
    sparse_topk_mlp,
    topk_relu,
)
from .tpu_flash_attention import (
    AttentionConfig,
    AttentionTiles,
    attention_tile_candidates,
    causal_attention,
    make_causal_attention,
    reference_causal_attention,
    select_attention_tiles,
)

__all__ = (
    "DEFAULT_VOCAB_TILE_SIZE",
    "AttentionConfig",
    "AttentionTiles",
    "FuzzyTopKCallable",
    "FuzzyTopKConfig",
    "SparseMlpConfig",
    "SparseMlpCallable",
    "attention_tile_candidates",
    "causal_attention",
    "fuzzy_topk_mlp",
    "fuzzy_topk_relu",
    "make_causal_attention",
    "make_mesh_fuzzy_topk_mlp",
    "make_mesh_sparse_topk_mlp",
    "naive_fuzzy_topk_mlp",
    "naive_dense_topk_mlp",
    "pallas_sparse_decode",
    "reference_causal_attention",
    "reference_sparse_decode",
    "select_attention_tiles",
    "sparse_topk_mlp",
    "tiled_tied_cross_entropy",
    "tiled_tied_cross_entropy_losses",
    "topk_relu",
)
