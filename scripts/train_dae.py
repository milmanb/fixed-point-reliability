"""Train a convolutional denoising autoencoder (proposal, Sec. 3).

    L = |f(y) - x|_1 + lambda_id * L_idem(f, f(y)),   y = x + sigma n,   sigma ~ U[sigma_min, sigma_max]

Training sees Gaussian noise only, with sigma in [0.05, 0.25] by default. At evaluation,
noise at sigma = 0.1 and 0.2 is in distribution, sigma = 0.3 and 0.5 are held-out levels,
and blur and masks are unseen operators. The last 5,000 training images are held out
for validation; the test set is used only by the evaluation scripts.

    python scripts/train_dae.py --lambda-id 0 --seed 0
"""

import argparse
import time

import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist
from fpr.losses import idempotence_loss, reconstruction_loss
from fpr.models import ConvAutoencoder, save_checkpoint

VAL_SIGMAS = (0.1, 0.2, 0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambda-id", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=5000)
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


def main():
    args = parse_args()
    name = f"dae_lam{args.lambda_id:g}_seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    images, _ = load_fashion_mnist("train")
    x_train = images[:-args.val_size].to(device)
    x_val = images[-args.val_size:].to(device)
    model = ConvAutoencoder(args.width, args.latent).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = -(-len(x_train) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"{name}: {n_params:,} parameters, {len(x_train)} training images, device {device}")

    history = []
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
            if args.lambda_id > 0:
                idem = idempotence_loss(model, fy)
                idem_sum += idem.detach()
                loss = loss + args.lambda_id * idem
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        row = {"epoch": epoch, "train_rec": rec_sum.item() / steps_per_epoch,
               "train_idem": idem_sum.item() / steps_per_epoch if args.lambda_id > 0 else float("nan"),
               **validate(model, x_val, args.seed), "seconds": time.perf_counter() - start}
        history.append(row)
        print(f"  epoch {epoch:>3}  rec {row['train_rec']:.4f}  val e@0.2 {row['val_e@0.2']:.4f}  "
              f"val g@0.2 {row['val_g@0.2']:.4f}  ({row['seconds']:.1f}s)", flush=True)

    checkpoint_dir = REPO_ROOT / args.checkpoints
    log_dir = REPO_ROOT / args.logs
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_dir / f"{name}.pt", model.cpu(), vars(args), history)
    pd.DataFrame(history).to_csv(log_dir / f"{name}.csv", index=False, float_format="%.6g")
    print(f"Saved {checkpoint_dir / f'{name}.pt'}")


if __name__ == "__main__":
    main()
