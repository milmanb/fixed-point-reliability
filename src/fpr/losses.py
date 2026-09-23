"""Training losses for the convolutional denoising autoencoders (proposal, Sec. 3).

    L = |f(y) - x|_1 + lambda_id * L_idem(f, f(y)),   lambda_id in {0, 0.1, 1}
"""

from torch.func import functional_call


def reconstruction_loss(fy, x):
    """Mean L1 distance between the restoration fy = f(y) and the clean image x."""
    return (fy - x).abs().mean()


def idempotence_loss(model, fy, routing="both"):
    """Penalty on the idempotence residual |f(f(y)) - f(y)|, averaged over pixels.

    fy = model(y) is the restoration from the reconstruction term, computed with gradients.

    `routing` selects which application of f receives parameter gradients:
      "both"  plain autograd through model(fy). This is the loss exactly as written in the
              proposal, with no stop-gradients. It lets the optimizer lower the residual by any
              route, including reshaping f around its own outputs, so it is the strongest form
              of explicit idempotence training. Default.
      "inner" as in IGN's idempotence term: the outer f uses detached parameters, so outputs
              are pulled toward the current fixed-point set without reshaping that set.
      "outer" the fixed-point set is reshaped around the current outputs, held constant.

    The gradient of "both" is exactly the sum of the gradients of "inner" and "outer"
    (tests/test_losses.py). On a trained lambda_id = 0 model its norm is 4.6 times that of the
    reconstruction gradient, against 1.8 for "inner" and 3.8 for "outer".
    """
    if routing not in ("both", "inner", "outer"):
        raise ValueError(f"routing must be 'both', 'inner' or 'outer', got {routing!r}")
    if routing == "outer":
        fy = fy.detach()
    if routing == "inner":
        frozen = {name: p.detach() for name, p in model.named_parameters()}
        outer = functional_call(model, frozen, (fy,))
    else:
        outer = model(fy)
    return (outer - fy).abs().mean()
