"""Train a convolutional denoising autoencoder (proposal, Sec. 3).

    L = |f(y) - x|_1 + lambda_id * L_idem(f, f(y)),   y = x + sigma n,   sigma ~ U[sigma_min, sigma_max]

Training sees Gaussian noise only, with sigma in [0.05, 0.25] by default. At evaluation,
noise at sigma = 0.1 and 0.2 is in distribution, sigma = 0.3 and 0.5 are held-out levels,
and blur and masks are unseen operators. The last 5,000 training images are held out
for validation; the test set is used only by the evaluation scripts.

With lambda_id = 1 from the first step, the network collapses within one epoch to a
constant output: the bottleneck model to the per-pixel median image (validation error 0.21,
g about 1e-6), the skip model (--architecture skip) to black (0.29, g about 1e-10).
--lambda-warmup ramps lambda_id linearly from 0 over the given number of epochs, so the
network learns to denoise before the idempotence term reaches full weight.

    python scripts/train_dae.py --lambda-id 0 --seed 0
    python scripts/train_dae.py --lambda-id 1 --lambda-warmup 5 --seed 0
"""

import argparse
import time

import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist
from fpr.losses import idempotence_loss, reconstruction_loss
from fpr.models import ARCHITECTURES, save_checkpoint

VAL_SIGMAS = (0.1, 0.2, 0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambda-id", type=float, default=0.0)
    parser.add_argument("--lambda-warmup", type=float, default=0.0,
                        help="epochs over which lambda_id rises linearly from 0 (default: no warm-up)")
    parser.add_argument("--routing", choices=("both", "inner", "outer"), default="both",
                        help="which application of f the idempotence term updates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.25)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="bottleneck")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for PyTorch; 3 per run suits three parallel runs on 8 cores")
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="checkpoint directory, relative to the repository")
    parser.add_argument("--logs", type=str, default="results/train",
                        help="training-log directory, relative to the repository")
    return parser.parse_args()


@torch.no_grad()
def validate(model, x_val, seed):
    """Mean error e and idempotence residual g on fixed noisy copies of the validation set."""
    model.eval()
    gen = torch.Generator(device=x_val.device).manual_seed(10_000 + seed)
    row = {}
    for sigma in VAL_SIGMAS:
        y = x_val + sigma * torch.randn(x_val.shape, generator=gen, device=x_val.device)
        fy = torch.cat([model(part) for part in y.split(1000)])
        ffy = torch.cat([model(part) for part in fy.split(1000)])
        row[f"val_e@{sigma:g}"] = (fy - x_val).abs().mean().item()
        row[f"val_g@{sigma:g}"] = (ffy - fy).abs().mean().item()
    model.train()
    return row


def lambda_at(step, lambda_id, warmup_steps):
    """Weight of the idempotence term at a training step: a linear ramp, then constant."""
    if warmup_steps <= 0:
        return lambda_id
    return lambda_id * min(1.0, step / warmup_steps)


def main():
    args = parse_args()
    warmup = f"w{args.lambda_warmup:g}" if args.lambda_warmup > 0 else ""
    routing = "" if args.routing == "both" else f"_{args.routing}"
    prefix = "dae" if args.architecture == "bottleneck" else "unet"
    name = f"{prefix}_lam{args.lambda_id:g}{warmup}{routing}_seed{args.seed}"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    images, _ = load_fashion_mnist("train")
    x_train = images[:-args.val_size].to(device)
    x_val = images[-args.val_size:].to(device)
    config = {"width": args.width}
    if args.architecture == "bottleneck":
        config["latent"] = args.latent
    model = ARCHITECTURES[args.architecture](**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = -(-len(x_train) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    warmup_steps = round(args.lambda_warmup * steps_per_epoch)
    checkpoint_dir = REPO_ROOT / args.checkpoints
    log_dir = REPO_ROOT / args.logs
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {n_params:,} parameters, {len(x_train)} training images, device {device}")

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        rec_sum = torch.zeros((), device=device)
        idem_sum = torch.zeros((), device=device)
        for idx in torch.randperm(len(x_train), generator=gen, device=device).split(args.batch_size):
            x = x_train[idx]
            sigma = args.sigma_min + (args.sigma_max - args.sigma_min) * torch.rand(
                (len(x), 1, 1, 1), generator=gen, device=device)
            y = x + sigma * torch.randn(x.shape, generator=gen, device=device)
            fy = model(y)
            loss = reconstruction_loss(fy, x)
            rec_sum += loss.detach()
            lam = lambda_at(step, args.lambda_id, warmup_steps)
            if args.lambda_id > 0:
                idem = idempotence_loss(model, fy, routing=args.routing)
                idem_sum += idem.detach()
                loss = loss + lam * idem
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
        row = {"epoch": epoch, "lambda": lam, "train_rec": rec_sum.item() / steps_per_epoch,
               "train_idem": idem_sum.item() / steps_per_epoch if args.lambda_id > 0 else float("nan"),
               **validate(model, x_val, args.seed), "seconds": time.perf_counter() - start}
        history.append(row)
        # Written every epoch, so an interrupted run keeps its training curve.
        pd.DataFrame(history).to_csv(log_dir / f"{name}.csv", index=False, float_format="%.6g")
        print(f"  epoch {epoch:>3}  lambda {lam:.2f}  rec {row['train_rec']:.4f}  "
              f"val e@0.2 {row['val_e@0.2']:.4f}  val g@0.2 {row['val_g@0.2']:.4f}  "
              f"({row['seconds']:.1f}s)", flush=True)

    save_checkpoint(checkpoint_dir / f"{name}.pt", model.cpu(), vars(args), history)
    print(f"Saved {name}.pt")


if __name__ == "__main__":
    main()
