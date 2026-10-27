"""Download the per-image dumps behind the committed results from the v1.0 GitHub release.

The analysis scripts read these dumps, which are git-ignored (186 MB in total). Recreating them
with scripts/evaluate_models.py, scripts/smoke_projectors.py and scripts/evaluate_pretrained.py
takes hours; with them in place, every analysis command in the README reproduces the committed
outputs. Each file is checked
against its SHA-256 digest before it is moved into place.

    python scripts/download_dumps.py
"""

import argparse

from fpr.data import REPO_ROOT
from fpr.download import fetch, sha256  # noqa: F401  (sha256 is part of this script's interface)

RELEASE = "https://github.com/milmanb/fixed-point-reliability/releases/download/v1.0"
DUMPS = {  # release asset -> (result folder, SHA-256 of the file)
    "models_per_image.csv.gz":
        ("results/models", "c09dd9ad14fe0077309d313a796a4a076e01b6b1c311f7b2cab48eda4726ec54"),
    "models_skip_per_image.csv.gz":
        ("results/models_skip", "90392271015404239040f6631d00190e2f8530328e69ec67b2149d859bb2ac3d"),
    "smoke_projectors_per_image.csv.gz":
        ("results/smoke_projectors", "5e51663a6bae249e020667c8270ce91f6c473af9e368880e62c40efb69ab8246"),
    "pretrained_per_image.csv.gz":
        ("results/pretrained", "28c240f0a4d26d0cf4f3113af979510718c1c3984f568e0db8e1ae42f4ddde55"),
    "pretrained_noclip_per_image.csv.gz":
        ("results/pretrained_noclip", "db5d3ce08adbf4778320fe61252ef921ddba690b47731984c4eea1594f0dfabd"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace dumps that differ from the release")
    parser.add_argument("--url", default=RELEASE, help="where the release assets are")
    return parser.parse_args()


def main():
    args = parse_args()
    for asset, (folder, expected) in DUMPS.items():
        target = REPO_ROOT / folder / "per_image.csv.gz"
        status = fetch(f"{args.url}/{asset}", target, expected, force=args.force)
        print(f"{folder}/per_image.csv.gz: {status}")


if __name__ == "__main__":
    main()
