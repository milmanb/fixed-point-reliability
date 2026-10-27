"""Checked downloads: a file is moved into place only if its SHA-256 digest matches."""

import hashlib
import shutil
import urllib.request


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
            raise SystemExit(f"{target} differs from the expected file; rerun with --force to replace it")
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
