"""Training losses for the convolutional denoising autoencoders (proposal, Sec. 3).

    L = |f(y) - x|_1 + lambda_id * L_idem(f, f(y)),   lambda_id in {0, 1}
"""


def reconstruction_loss(fy, x):
    """Mean L1 distance between the restoration fy = f(y) and the clean image x."""
    return (fy - x).abs().mean()


def idempotence_loss(model, fy):
    """Penalty on the idempotence residual |f(f(y)) - f(y)|, averaged over pixels.

    fy = model(y) is the restoration from the reconstruction term, computed with gradients.

    Which application of f receives parameter gradients is a design choice:
      (a) both applications (plain autograd through model(fy))  <- used here;
      (b) inner only, as in IGN's idempotence term: the outer f uses detached
          parameters, so outputs are pulled toward the current fixed-point set
          without reshaping that set;
      (c) outer only: the fixed-point set is reshaped around the current outputs,
          which are treated as constants (fy.detach()).
    The gradient of (a) is exactly the sum of the gradients of (b) and (c). On a trained
    lambda_id = 0 model its norm is 4.6 times that of the reconstruction gradient,
    against 1.8 for (b) and 3.8 for (c).

    For (b), a frozen outer application (gradients still flow into its input) is
        frozen = {name: p.detach() for name, p in model.named_parameters()}
        outer = torch.func.functional_call(model, frozen, (fy,))
    """
    # Option (a) is the loss exactly as written in the approved proposal, with no
    # stop-gradients. It lets the optimizer lower the residual by any route, including
    # reshaping f around its own outputs. This is the strongest form of explicit
    # idempotence training, so it is the most direct test of whether such training
    # decouples the residual g from the true error.
    return (model(fy) - fy).abs().mean()
