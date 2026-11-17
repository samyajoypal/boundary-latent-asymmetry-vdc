from __future__ import annotations

import itertools

import torch
from torch import Tensor


def cluster_purity(y_true: Tensor, y_pred: Tensor) -> float:
    y_true = y_true.detach().cpu().long()
    y_pred = y_pred.detach().cpu().long()
    correct = 0
    for cluster in torch.unique(y_pred):
        mask = y_pred == cluster
        if mask.any():
            counts = torch.bincount(y_true[mask])
            correct += int(counts.max())
    return correct / max(int(y_true.numel()), 1)


def best_map_accuracy(y_true: Tensor, y_pred: Tensor, max_bruteforce_classes: int = 8) -> float:
    y_true = y_true.detach().cpu().long()
    y_pred = y_pred.detach().cpu().long()
    true_classes = torch.unique(y_true).tolist()
    pred_classes = torch.unique(y_pred).tolist()
    if len(true_classes) != len(pred_classes) or len(true_classes) > max_bruteforce_classes:
        return cluster_purity(y_true, y_pred)

    true_to_idx = {label: idx for idx, label in enumerate(true_classes)}
    pred_to_idx = {label: idx for idx, label in enumerate(pred_classes)}
    y_true_idx = torch.tensor([true_to_idx[int(v)] for v in y_true])
    y_pred_idx = torch.tensor([pred_to_idx[int(v)] for v in y_pred])

    best = 0
    for perm in itertools.permutations(range(len(true_classes))):
        mapped = torch.tensor([perm[int(v)] for v in y_pred_idx])
        best = max(best, int((mapped == y_true_idx).sum()))
    return best / max(int(y_true.numel()), 1)
