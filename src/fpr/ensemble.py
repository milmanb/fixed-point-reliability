"""Ensemble disagreement as an epistemic baseline for reliability ranking.

For a seed group {f_1, ..., f_M} and the mean prediction bar_f = (1/M) sum_i f_i(y),

    dis(y) = (1/(M |Omega|)) sum_i ||f_i(y) - bar_f(y)||_1

attached to every member's per-image rows. Self-consistency (the idempotence residual)
tracks aleatoric uncertainty and can collapse on unseen operators; cross-model disagreement
targets the epistemic part (cf. arXiv:2604.17112).
"""

import time

import numpy as np
import torch


def model_group(name):
    """dae_lam0_seed2 -> dae_lam0; projectors and single-seed names keep their name."""
    return name.rsplit("_seed", 1)[0]


def seed_groups(models):
    """Map group name -> list of (member_name, callable) for groups with at least 2 seeds."""
    groups = {}
    for name, f in models.items():
        if "_seed" not in name:
            continue
        groups.setdefault(model_group(name), []).append((name, f))
    return {g: members for g, members in groups.items() if len(members) >= 2}


@torch.no_grad()
def disagreement(members, y, batch_size=2048):
    """Per-image mean absolute disagreement of `members` about y.

    members: sequence of callables. Returns an (n,) float64 array.
    """
    if len(members) < 2:
        raise ValueError("disagreement needs at least two models")
    preds = []
    for f in members:
        chunks = [f(part) for part in y.split(batch_size)]
        preds.append(torch.cat(chunks))
    stacked = torch.stack(preds, dim=0)  # (M, N, ...)
    mean = stacked.mean(0, keepdim=True)
    return (stacked - mean).abs().flatten(2).mean((0, 2)).detach().cpu().double().numpy()


def attach_disagreement(per_image, models, x, seed=0, conditions=None, verbose=True,
                        batch_size=2048):
    """Add a `dis` column to `per_image` for every seed-group that has at least two members.

    Re-observes each condition with the same seed as evaluation, so the y's match.
    Rows whose model is not in a multi-seed group keep dis = NaN.
    """
    from fpr.evaluation import CONDITIONS, observe

    conditions = conditions or CONDITIONS
    groups = seed_groups(models)
    per_image = per_image.copy()
    per_image["dis"] = np.nan
    if not groups:
        return per_image

    for condition in conditions:
        y, _ = observe(condition, x, seed)
        for group, members in groups.items():
            start = time.perf_counter()
            values = disagreement([f for _, f in members], y, batch_size=batch_size)
            names = {name for name, _ in members}
            for name in names:
                member_mask = ((per_image["condition"] == condition.label)
                               & (per_image["model"] == name))
                order = per_image.loc[member_mask, "image"].to_numpy()
                per_image.loc[member_mask, "dis"] = values[order]
            if verbose:
                print(f"  {condition.label:<22} dis/{group:<16} mean={values.mean():.4f}  "
                      f"({time.perf_counter() - start:.1f}s)", flush=True)
    return per_image
