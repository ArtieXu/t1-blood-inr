#!/usr/bin/env python3
"""
deps.py -- Gate A dependency preflight.

verify_migration.py --env checks torch, torchkbnufft and tinycudann only.
train_inr_unsup_spiral.py additionally imports utils.py, which imports
torchvision, skimage (scikit-image), imageio and matplotlib.  A missing one of
those raises ImportError in the first second of a run, after the environment
has already been declared "verified".  This closes that gap before any GPU time
is spent.

    python3 deps.py            # exit 0 = every import in the
                                            # Gate-B code path succeeded
"""
from __future__ import annotations

import importlib
import sys

# name -> (pip package, imported by)
REQUIRED = {
    "torch":        ("torch==2.7.1 (cu128 index)", "train/model/utils/spiral_nufft"),
    "torchvision":  ("torchvision (cu128 index, matched to torch)", "utils.make_grid"),
    "torchkbnufft": ("torchkbnufft==1.5.2", "spiral_nufft, utils"),
    "tinycudann":   ("git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch",
                     "model.INR"),
    "numpy":        ("numpy", "train, load_spiral, utils"),
    "scipy":        ("scipy", "train (io.savemat), load_spiral"),
    "h5py":         ("h5py", "load_spiral, utils"),
    "matplotlib":   ("matplotlib", "load_spiral, utils"),
    "imageio":      ("imageio", "utils.visual_mag"),
    "skimage":      ("scikit-image", "utils.metrics"),
    "tqdm":         ("tqdm", "train, model"),
}
# imported only when the flag is used; absence is not a Gate A failure
OPTIONAL = {
    "wandb":  ("wandb", "train --wandb (deliberately unused on 5070)"),
    "pandas": ("pandas", "analysis scripts only"),
}


def probe(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:                                  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "version unknown"))


def main() -> int:
    failures: list[str] = []
    print("required (Gate-B code path):")
    for name, (pkg, who) in REQUIRED.items():
        ok, detail = probe(name)
        print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} {detail:34s} <- {who}")
        if not ok:
            failures.append(f"{name}  (pip install {pkg})")

    print("\noptional:")
    for name, (pkg, who) in OPTIONAL.items():
        ok, detail = probe(name)
        print(f"  {'ok  ' if ok else '--  '} {name:14s} {detail:34s} <- {who}")

    # torch and torchvision must come from the same CUDA build family, or
    # torchvision's C++ extension fails to load with an opaque error at first use.
    #
    # CAREFUL: the two libraries report CUDA differently and are NOT directly
    # comparable --  torch.version.cuda is a dotted string ('12.8') while
    # torchvision.version.cuda is a CUDA_VERSION integer (12080 = 12*1000+8*10).
    # Comparing them raw always "fails". Compare the +cuXXX local version tags
    # instead, and fall back to a normalized (major, minor) pair.
    def _cu_tag(version_string):
        """'2.7.1+cu128' -> 'cu128'"""
        return version_string.partition("+")[2] or None

    def _cu_pair(value):
        """'12.8' -> (12, 8);  12080 / '12080' -> (12, 8)"""
        if value is None:
            return None
        text = str(value)
        if "." in text:
            bits = text.split(".")
            return (int(bits[0]), int(bits[1]) if len(bits) > 1 else 0)
        if text.isdigit():
            n = int(text)
            return (n // 1000, (n % 1000) // 10)
        return None

    try:
        import torch, torchvision                              # noqa: E401
        t_tag, v_tag = _cu_tag(torch.__version__), _cu_tag(torchvision.__version__)
        t_pair = _cu_pair(torch.version.cuda)
        v_pair = _cu_pair(getattr(torchvision.version, "cuda", None))
        print(f"\ntorch {torch.__version__} (cuda {torch.version.cuda} -> {t_pair})")
        print(f"torchvision {torchvision.__version__} "
              f"(cuda {getattr(torchvision.version, 'cuda', None)} -> {v_pair})")
        if t_tag and v_tag:
            verdict = "match" if t_tag == v_tag else "MISMATCH"
            print(f"build tags: torch {t_tag} / torchvision {v_tag}  -> {verdict}")
            if t_tag != v_tag:
                failures.append(
                    f"torchvision build {v_tag} != torch build {t_tag}; install "
                    f"torchvision from the https://download.pytorch.org/whl/{t_tag} index")
        elif t_pair and v_pair and t_pair != v_pair:
            failures.append(
                f"torchvision CUDA {v_pair[0]}.{v_pair[1]} != torch CUDA "
                f"{t_pair[0]}.{t_pair[1]}; reinstall torchvision from the matching index")
        # a real load of the compiled extension -- the thing a version check cannot prove
        try:
            from torchvision.utils import make_grid
            make_grid(torch.zeros(2, 1, 4, 4))
            print("torchvision.utils.make_grid ran (the call utils.py actually makes)")
        except Exception as exc:                               # noqa: BLE001
            failures.append(f"torchvision imported but make_grid failed: "
                            f"{type(exc).__name__}: {exc}")
    except Exception:                                          # noqa: BLE001
        pass

    if failures:
        print("\nPREFLIGHT FAILED:")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nPREFLIGHT PASSED: every import in the Gate-B code path resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
