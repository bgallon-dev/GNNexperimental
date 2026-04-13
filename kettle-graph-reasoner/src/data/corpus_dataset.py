r"""Corpus dataset for tier-N NPZ graphs.

Each NPZ bundles one graph (``x``, ``edge_index``, ``edge_attr``,
``schema_*``) plus ``n_tasks`` tasks (``task_j_*``). A training example is
a single (graph, task) pair — the model consumes one graph at a time.

Edge types are recovered from the first 25 dims of ``edge_attr`` (a
one-hot over edge-type slots baked in by ``feature_encoder.encode_edges``).
The schema edge descriptor is assembled from the padded ``schema_*``
arrays into a (MAX_EDGE_TYPES=30, 13)-shaped feature matrix; the node
descriptor is a (MAX_NODE_TYPES=16, 4) one-hot over layer assignment.
These dims define the model's ``edge_feat_dim`` / ``node_feat_dim_schema``
at construction time, so read them from the dataset after instantiation.

Graph-level splits are deterministic under ``split_seed`` — tasks within
a graph stay together to avoid leaking topology across splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


MAX_EDGE_TYPES = 30
MAX_NODE_TYPES = 16
NUM_LAYERS = 4          # source=0, claim=1, entity=2, auxiliary=3
EDGE_TYPE_ONEHOT_DIMS = 25  # first 25 dims of edge_attr
EDGE_DESC_DIM = 1 + 1 + NUM_LAYERS + NUM_LAYERS + 4  # cat_onehot + directed + src + tgt + cat_flag (see below)
# Composition per edge-type slot:
#   [category_onehot(4) | directed(1) | source_layers(4) | target_layers(4)] = 13
EDGE_DESC_DIM = 4 + 1 + NUM_LAYERS + NUM_LAYERS  # 13
NODE_DESC_DIM = NUM_LAYERS  # one-hot over 4 layers = 4
NUM_EDGE_CATEGORIES = 4


@dataclass
class Sample:
    x: Tensor                  # (N, 32)
    edge_index: Tensor         # (2, E) long
    edge_type: Tensor          # (E,) long
    edge_descriptor: Tensor    # (MAX_EDGE_TYPES, EDGE_DESC_DIM)
    node_descriptor: Tensor    # (MAX_NODE_TYPES, NODE_DESC_DIM)
    query: Tensor              # (9,)
    labels: Tensor             # (N,) in [0, 1]
    task_type: int


def _onehot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    valid = (idx >= 0) & (idx < num_classes)
    out[np.arange(idx.shape[0])[valid], idx[valid]] = 1.0
    return out


def _build_graph_tensors(npz: np.lib.npyio.NpzFile) -> dict:
    x = torch.from_numpy(npz["x"].astype(np.float32))
    edge_index = torch.from_numpy(npz["edge_index"].astype(np.int64))
    edge_attr = npz["edge_attr"].astype(np.float32)
    edge_type = torch.from_numpy(
        edge_attr[:, :EDGE_TYPE_ONEHOT_DIMS].argmax(axis=1).astype(np.int64)
    )

    cat = npz["schema_edge_category"].astype(np.int64)           # (30,)
    directed = npz["schema_edge_directed"].astype(np.float32)    # (30,)
    src_layers = npz["schema_edge_source_layers"].astype(np.float32)  # (30, 4)
    tgt_layers = npz["schema_edge_target_layers"].astype(np.float32)  # (30, 4)
    cat_oh = _onehot(np.clip(cat, -1, NUM_EDGE_CATEGORIES - 1), NUM_EDGE_CATEGORIES)
    edge_descriptor = np.concatenate(
        [cat_oh, directed[:, None], src_layers, tgt_layers], axis=1
    ).astype(np.float32)  # (30, 13)

    node_layer = npz["schema_node_layer_assignment"].astype(np.int64)  # (16,)
    node_descriptor = _onehot(
        np.clip(node_layer, -1, NUM_LAYERS - 1), NUM_LAYERS
    )  # (16, 4)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_descriptor": torch.from_numpy(edge_descriptor),
        "node_descriptor": torch.from_numpy(node_descriptor),
    }


def _build_task_tensors(npz: np.lib.npyio.NpzFile, j: int) -> dict:
    return {
        "query": torch.from_numpy(npz[f"task_{j}_query"].astype(np.float32)),
        "labels": torch.from_numpy(npz[f"task_{j}_labels"].astype(np.float32)),
        "task_type": int(npz[f"task_{j}_type"]),
    }


def _split_files(files: list[Path], split: str, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    if split == "train":
        sel = idx[:n_train]
    elif split == "val":
        sel = idx[n_train : n_train + n_val]
    elif split == "test":
        sel = idx[n_train + n_val :]
    else:
        raise ValueError(f"unknown split {split!r}")
    return [files[i] for i in sel]


class CorpusDataset(Dataset):
    def __init__(
        self,
        corpus_dir: Path | str,
        split: str = "train",
        split_seed: int = 0,
        cache_in_memory: bool = True,
    ) -> None:
        corpus_dir = Path(corpus_dir)
        files = sorted(corpus_dir.glob("graph_*.npz"))
        if not files:
            raise FileNotFoundError(f"no graph_*.npz under {corpus_dir}")
        self.files = _split_files(files, split, split_seed)
        self.cache_in_memory = cache_in_memory
        self._graph_cache: dict[int, dict] = {}

        # Build flat index of (graph_idx, task_idx) pairs. n_tasks is a
        # scalar in each NPZ — cheap to read up-front.
        self.index: list[tuple[int, int]] = []
        for gi, f in enumerate(self.files):
            with np.load(f) as npz:
                n_tasks = int(npz["n_tasks"])
            for j in range(n_tasks):
                self.index.append((gi, j))

        # Expose dims for model construction.
        self.node_feat_dim = 32
        self.edge_feat_dim_schema = EDGE_DESC_DIM
        self.node_feat_dim_schema = NODE_DESC_DIM
        self.query_dim = 9
        self.num_edge_types_max = MAX_EDGE_TYPES

    def __len__(self) -> int:
        return len(self.index)

    def _get_graph(self, gi: int) -> dict:
        if self.cache_in_memory and gi in self._graph_cache:
            return self._graph_cache[gi]
        with np.load(self.files[gi]) as npz:
            graph = _build_graph_tensors(npz)
        if self.cache_in_memory:
            self._graph_cache[gi] = graph
        return graph

    def __getitem__(self, i: int) -> Sample:
        gi, j = self.index[i]
        graph = self._get_graph(gi)
        with np.load(self.files[gi]) as npz:
            task = _build_task_tensors(npz, j)
        return Sample(
            x=graph["x"],
            edge_index=graph["edge_index"],
            edge_type=graph["edge_type"],
            edge_descriptor=graph["edge_descriptor"],
            node_descriptor=graph["node_descriptor"],
            query=task["query"],
            labels=task["labels"],
            task_type=task["task_type"],
        )


def collate_single(batch: list[Sample]) -> Sample:
    """DataLoader collate for batch_size=1. Unwraps the single element."""
    if len(batch) != 1:
        raise ValueError("CorpusDataset only supports batch_size=1")
    return batch[0]
