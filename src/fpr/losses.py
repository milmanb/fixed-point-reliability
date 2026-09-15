"""Training losses for the convolutional denoising autoencoders (proposal, Sec. 3).

    L = |f(y) - x|_1 + lambda_id * L_idem(f, f(y)),   lambda_id in {0, 1}
"""

from torch.func import functional_call  # noqa: F401  (used by option (b) below)


def reconstruction_loss(fy, x):
    """Mean L1 distance between the restoration fy = f(y) and the clean image x."""
    return (fy - x).abs().mean()


def idempotence_loss(model, fy):
    """Penalty on the idempotence residual |f(f(y)) - f(y)|, averaged over pixels.

    fy = model(y) is the restoration from the reconstruction term, computed with gradients.

    The design choice is which application of f receives parameter gradients:
      (a) both applications (plain autograd through model(fy));
      (b) inner only, as in IGN's idempotence term: the outer f uses detached
          parameters, so outputs are pulled toward the current fixed-point set
          without reshaping that set;
      (c) outer only: the fixed-point set is reshaped around the current outputs,
          which are treated as constants (fy.detach()).

    A frozen outer application (gradients still flow into its input) can be written as
        frozen = {name: p.detach() for name, p in model.named_parameters()}
        outer = functional_call(model, frozen, (fy,))
    """
    # TODO: implement one of (a)-(c) and state in a comment why it fits our question.
    raise NotImplementedError("idempotence_loss is not implemented yet; see docstring")
