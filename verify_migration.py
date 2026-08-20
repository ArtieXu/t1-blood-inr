#!/usr/bin/env python3
"""Verify the minimum RTX 5070 migration set without third-party packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MIGRATION_5070_MANIFEST.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_pair(value: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else (0, 0)


def check_files() -> list[str]:
    errors: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        for row in rows:
            if row["required"] != "yes" or row["location"] != "local":
                continue
            path = ROOT / row["path"]
            if not path.is_file():
                errors.append(f"missing: {row['path']}")
                continue
            actual_size = path.stat().st_size
            if actual_size != int(row["bytes"]):
                errors.append(
                    f"size mismatch: {row['path']} "
                    f"({actual_size} != {row['bytes']})"
                )
                continue
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                errors.append(f"SHA-256 mismatch: {row['path']}")
    return errors


def check_environment() -> list[str]:
    errors: list[str] = []
    try:
        import torch
    except ImportError:
        return ["missing Python package: torch"]

    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    if version_pair(torch.__version__) < (2, 7):
        errors.append("PyTorch must be 2.7 or newer for the Blackwell baseline")
    if version_pair(torch.version.cuda or "") < (12, 8):
        errors.append("PyTorch CUDA runtime must be 12.8 or newer")
    if not torch.cuda.is_available():
        errors.append("CUDA is not available to PyTorch")
        return errors

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"gpu={name} capability={capability} vram_gib={total_gib:.2f}")
    if capability < (12, 0):
        errors.append(f"expected Blackwell capability >= (12, 0), got {capability}")
    if total_gib < 11:
        errors.append(f"expected approximately 12 GiB VRAM, got {total_gib:.2f}")

    for package in ("torchkbnufft", "tinycudann"):
        try:
            module = __import__(package)
            print(f"{package}={getattr(module, '__version__', 'unknown')}")
        except ImportError:
            errors.append(f"missing Python package: {package}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", action="store_true", help="also verify the RTX 5070 Python/CUDA stack"
    )
    args = parser.parse_args()

    errors = check_files()
    if args.env:
        errors.extend(check_environment())
    if errors:
        print("Migration verification FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Migration verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
