import pytest
import torch

from fpr.projectors import PCA, Identity, NearestNeighbor, Radial, principal_components


@pytest.fixture(scope="module")
def data():
    gen = torch.Generator().manual_seed(0)
    train = torch.rand((300, 1, 6, 6), generator=gen, dtype=torch.float64)
    queries = torch.randn((50, 1, 6, 6), generator=gen, dtype=torch.float64)
    return train, queries


def test_projectors_are_idempotent(data):
    train, queries = data
    mean, eigvecs, _ = principal_components(train)
    projectors = [Identity(), Radial.fit(train), PCA.from_components(mean, eigvecs, 5),
                  NearestNeighbor(train, device="cpu", chunk=16)]
    for projector in projectors:
        once = projector(queries)
        assert once.shape == queries.shape
        assert torch.allclose(projector(once), once, rtol=0, atol=1e-12), projector.name


def test_radial_output_lies_on_sphere(data):
    train, queries = data
    projector = Radial.fit(train)
    dist = (projector(queries).flatten(1) - projector.center).norm(dim=1)
    assert torch.allclose(dist, torch.full_like(dist, projector.radius))


def test_pca_residual_is_orthogonal_to_subspace(data):
    train, queries = data
    mean, eigvecs, eigvals = principal_components(train)
    projector = PCA.from_components(mean, eigvecs, 5)
    residual = (queries - projector(queries)).flatten(1)
    assert torch.allclose(residual @ projector.basis, torch.zeros(50, 5, dtype=torch.float64),
                          atol=1e-12)
    assert (eigvals[:-1] >= eigvals[1:] - 1e-12).all()


def test_nearest_neighbor_matches_brute_force(data):
    train, queries = data
    projector = NearestNeighbor(train, device="cpu", chunk=16, candidates=4)
    assert torch.equal(projector.indices(train), torch.arange(len(train)))
    brute = torch.cdist(queries.flatten(1), train.flatten(1)).argmin(1)
    assert torch.equal(projector.indices(queries), brute)
