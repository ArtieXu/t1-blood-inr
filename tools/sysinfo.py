#!/usr/bin/env python3
"""
sysinfo.py -- record the complete 5070 runtime signature.

Gate B requires "记录完整 runtime signature: GPU、驱动、PyTorch、CUDA、
tiny-cuda-nn、Python 版本".  This writes that as JSON so any later run can be
compared field by field, and so a T4-vs-5070 comparison can be rejected
automatically when the stack differs.

    python3 sysinfo.py                 # print to stdout
    python3 sysinfo.py -o sig.json     # also write a file

Exit code is always 0: this records, it does not gate.  Gate A is
doctor.sh.
"""
from __future__ import annotations

import argparse, importlib, json, os, platform, subprocess, sys, datetime

# every third-party module the Gate-B code path imports, derived from the
# import graph of train_inr_unsup_spiral.py -> {model, utils, load_spiral,
# spiral_nufft}.  wandb is listed but only imported when --wandb is passed.
CODE_PATH_MODULES = [
    "torch", "torchvision", "torchkbnufft", "tinycudann",
    "numpy", "scipy", "h5py", "matplotlib", "imageio", "skimage",
    "tqdm", "pandas",
]
OPTIONAL_MODULES = ["wandb"]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:                      # noqa: BLE001
        return f"IMPORT FAILED: {type(exc).__name__}: {exc}"
    for attr in ("__version__", "version", "VERSION"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    try:
        from importlib.metadata import version as md_version
        return md_version({"skimage": "scikit-image",
                           "tinycudann": "tinycudann"}.get(name, name))
    except Exception:                             # noqa: BLE001
        return "unknown"


def _tcnn_provenance() -> dict:
    """tiny-cuda-nn exposes no commit hash; record what can be recorded."""
    info: dict = {"version": _version("tinycudann")}
    try:
        import tinycudann as tcnn
        info["module_file"] = getattr(tcnn, "__file__", None)
        pkg = os.path.dirname(info["module_file"] or "")
        info["package_dir"] = pkg or None
        try:
            from importlib.metadata import metadata
            meta = metadata("tinycudann")
            info["dist_metadata"] = {k: meta[k] for k in ("Name", "Version", "Home-page")
                                     if k in meta}
        except Exception:                         # noqa: BLE001
            pass
    except Exception as exc:                      # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    info["expected_commit_from_MIGRATION_5070_md"] = \
        "749dd70c5afc5a9dadb85e5652ed65d55e0ba187"
    info["note"] = ("tinycudann does not expose its source commit at runtime. "
                    "Record the commit you actually built from by hand; a "
                    "different commit defines a new runtime branch and "
                    "invalidates cross-run comparison.")
    return info


def collect() -> dict:
    sig: dict = {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "is_wsl": "microsoft" in platform.release().lower(),
        },
        "env_vars": {k: os.environ.get(k) for k in (
            "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
            "LD_LIBRARY_PATH", "WANDB_MODE", "WANDB_DISABLED")},
        "nvidia_smi": _run(["nvidia-smi",
                            "--query-gpu=name,driver_version,memory.total,compute_cap",
                            "--format=csv,noheader"]),
        "nvcc": _run(["nvcc", "--version"]),
        "packages": {m: _version(m) for m in CODE_PATH_MODULES},
        "optional_packages": {m: _version(m) for m in OPTIONAL_MODULES},
        "tiny_cuda_nn": _tcnn_provenance(),
    }
    try:
        import torch
        sig["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            sig["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_gib": round(props.total_memory / 2 ** 30, 3),
                "multi_processor_count": props.multi_processor_count,
            }
    except Exception as exc:                      # noqa: BLE001
        sig["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    return sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", help="also write the signature to this JSON path")
    args = ap.parse_args()
    sig = collect()
    text = json.dumps(sig, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
