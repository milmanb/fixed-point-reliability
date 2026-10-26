"""Download the per-image dumps behind the committed results from the v1.0 GitHub release.

The analysis scripts read these dumps, which are git-ignored (185 MB in total). Recreating them
with scripts/evaluate_models.py and scripts/smoke_projectors.py takes hours; with them in place,
every analysis command in the README reproduces the committed outputs. Each file is checked
against its SHA-256 digest before it is moved into place.

    python scripts/download_dumps.py
"""

import argparse
import hashlib
import shutil
import urllib.request

from fpr.data import REPO_ROOT

RELEASE = "https://github.com/milmanb/fixed-point-reliability/releases/download/v1.0"
DUMPS = {  # release asset -> (result folder, SHA-256 of the file)
    "models_per_image.csv.gz":
        ("results/models", "c09dd9ad14fe0077309d313a796a4a076e01b6b1c311f7b2cab48eda4726ec54"),
    "models_skip_per_image.csv.gz":
        ("results/models_skip", "90392271015404239040f6631d00190e2f8530328e69ec67b2149d859bb2ac3d"),
    "smoke_projectors_per_image.csv.gz":
        ("results/smoke_projectors", "5e51663a6bae249e020667c8270ce91f6c473af9e368880e62c40efb69ab8246"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace dumps that differ from the release")
    parser.add_argument("--url", default=RELEASE, help="where the release assets are")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, target, expected, force=False):
    """Download `url` to `target` unless an identical file is there; returns what was done."""
    if target.exists():
        if sha256(target) == expected:
            return "present"
        if not force:
            raise SystemExit(f"{target} differs from the release; rerun with --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as file:
            shutil.copyfileobj(response, file)
    except OSError as error:  # URLError, HTTPError and timeouts are all OSErrors
        partial.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {error}") from None
    if sha256(partial) != expected:
        partial.unlink()
        raise SystemExit(f"{url} does not match its SHA-256 digest")
    partial.replace(target)
    return "downloaded"


def main():
    args = parse_args()
    for asset, (folder, expected) in DUMPS.items():
        target = REPO_ROOT / folder / "per_image.csv.gz"
        status = fetch(f"{args.url}/{asset}", target, expected, force=args.force)
        print(f"{folder}/per_image.csv.gz: {status}")


if __name__ == "__main__":
    main()
