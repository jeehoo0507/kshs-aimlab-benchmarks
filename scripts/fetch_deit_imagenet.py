"""Cache official ImageNet-pretrained, non-distilled DeiT weights with PyTorch."""

import argparse
import hashlib
from pathlib import Path

import torch


URLS = {
    "tiny": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    "small": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    "base": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=URLS.keys(), help="DeiT sizes to cache")
    args = parser.parse_args()

    for name in dict.fromkeys(args.models):
        url = URLS[name]
        filename = url.rsplit("/", 1)[-1]
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        expected_prefix = filename.rsplit("-", 1)[-1].removesuffix(".pth")
        if path.exists() and not sha256(path).startswith(expected_prefix):
            path.unlink()
        torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True, weights_only=True
        )
        digest = sha256(path)
        if not digest.startswith(expected_prefix):
            path.unlink()
            raise RuntimeError(f"SHA-256 mismatch for {filename}; invalid cache removed")
        print(f"{name}: {path}\n  SHA-256: {digest}")


if __name__ == "__main__":
    main()
