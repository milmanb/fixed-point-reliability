import numpy as np
import pandas as pd
import torch

from fpr.evaluation import CONDITIONS, per_image_signals, score
from fpr.models import ConvAutoencoder, Restorer
from fpr.projectors import PCA, Radial, principal_components


def test_process_pool_reproduces_serial_results():
    gen = torch.Generator().manual_seed(0)
    train = torch.rand((200, 1, 28, 28), generator=gen, dtype=torch.float64)
    x = torch.rand((60, 1, 28, 28), generator=gen, dtype=torch.float64)
    labels = torch.randint(0, 10, (60,), generator=gen)
    mean, eigvecs, _ = principal_components(train)
    torch.manual_seed(0)
    models = {"radial": Radial.fit(train), "pca8": PCA.from_components(mean, eigvecs, 8),
              "tiny": Restorer(ConvAutoencoder(width=4, latent=4), name="tiny", device="cpu")}
    conditions = [CONDITIONS[1], CONDITIONS[5], CONDITIONS[12]]  # noise, blur, box mask
    kwargs = dict(labels=labels, seed=3, probes=2, conditions=conditions, verbose=False)

    serial = per_image_signals(models, x, **kwargs)
    pooled = per_image_signals(models, x, jobs=2, threads=1, **kwargs)
    assert serial[["condition", "model", "image"]].equals(pooled[["condition", "model", "image"]])
    numeric = serial.select_dtypes("number").columns
    assert np.allclose(serial[numeric], pooled[numeric], rtol=1e-5, atol=1e-7, equal_nan=True)

    signals = ("b", "g", "d", "r_A", "div", "sure")
    pd.testing.assert_frame_equal(score(serial, signals, n_boot=20, seed=1),
                                  score(serial, signals, n_boot=20, seed=1, jobs=2))
