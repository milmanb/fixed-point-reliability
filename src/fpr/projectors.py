"""Exact projectors: maps P with P(P(y)) = P(y) for every input.

They serve as model-agnostic references for the learned models. For any exact
projector the idempotence residual g(y) is zero everywhere, so g cannot rank
images by error. What differs between projectors is the fixed-point set
(the range of P) and hence how informative displacement is:

    identity   range = all images            (loosest; d = 0 as well)
    radial     range = a sphere around the training mean
    pca{k}     range = a k-dimensional affine subspace
    nn         range = the training images   (tightest; outputs are real images)

Inputs and outputs are image batches of shape (N, C, H, W).
"""

import torch


class Identity:
    name = "identity"

    def __call__(self, y):
        return y


class Radial:
    """Euclidean projection onto the sphere ||v - c||_2 = R.

    c is the training mean and R the mean distance of training images from it.
    """

    name = "radial"

    def __init__(self, center, radius):
        self.center = center
        self.radius = float(radius)

    @classmethod
    def fit(cls, x_train):
        v = x_train.flatten(1)
        center = v.mean(0)
        return cls(center, (v - center).norm(dim=1).mean())

    def __call__(self, y):
        center = self.center.to(y)
        v = y.flatten(1) - center
        norm = v.norm(dim=1, keepdim=True).clamp_min(torch.finfo(y.dtype).tiny)
        return (center + self.radius * v / norm).view_as(y)


def principal_components(x_train):
    """Mean, eigenvectors (columns, by decreasing variance) and eigenvalues of the training data."""
    v = x_train.flatten(1).to(torch.float64)
    mean = v.mean(0)
    centered = v - mean
    cov = centered.T @ centered / (v.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return mean, eigvecs.flip(1), eigvals.flip(0).clamp_min(0)


class PCA:
    """Orthogonal projection onto the affine span of the top-k principal directions."""

    def __init__(self, mean, basis):
        self.mean = mean
        self.basis = basis
        self.name = f"pca{basis.shape[1]}"

    @classmethod
    def from_components(cls, mean, eigvecs, k):
        return cls(mean, eigvecs[:, :k].contiguous())

    def __call__(self, y):
        mean, basis = self.mean.to(y), self.basis.to(y)
        v = y.flatten(1) - mean
        return (mean + (v @ basis) @ basis.T).view_as(y)


class NearestNeighbor:
    """Projection onto a finite image bank: return the closest bank image in L2.

    Distances are screened in float32 on the GPU (if present) and the top
    candidates are re-ranked exactly in float64. Without the exact re-ranking,
    float32 round-off could map a bank image to a near-duplicate, breaking
    idempotence for reasons unrelated to the projector.
    """

    name = "nn"

    def __init__(self, bank, device=None, chunk=512, candidates=8):
        self.bank = bank.flatten(1).to("cpu", torch.float64)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bank32 = self.bank.to(self.device, torch.float32)
        self.bank32_sq = self.bank32.pow(2).sum(1)
        self.chunk = chunk
        self.candidates = candidates

    @torch.no_grad()
    def indices(self, y):
        queries = y.flatten(1).to("cpu", torch.float64)
        found = []
        for part in queries.split(self.chunk):
            part32 = part.to(self.device, torch.float32)
            # ||q||^2 is constant per row, so it does not change the ranking.
            dist = self.bank32_sq[None, :] - 2.0 * part32 @ self.bank32.T
            cand = dist.topk(self.candidates, dim=1, largest=False).indices.cpu()
            exact = (self.bank[cand] - part[:, None, :]).pow(2).sum(-1)
            found.append(cand.gather(1, exact.argmin(1, keepdim=True)).squeeze(1))
        return torch.cat(found)

    def __call__(self, y):
        return self.bank[self.indices(y)].to(y).view_as(y)
