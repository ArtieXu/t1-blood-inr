#!/usr/bin/env python3
"""
run.py -- run the REAL train_inr_unsup_spiral.py end to end on a CPU.

Everything is the project's own code except two substitutions, both printed at
startup so they can never be mistaken for a real run:

  1. tinycudann      -> a shape-faithful CPU stub (stub/tinycudann.py)
  2. INR.build_pos   -> same function with device='cuda' relaxed to the CPU

Purpose: prove that every non-tcnn code path -- argparse validation, h5py load,
DCF, SpiralNUFFT, the scale quantile, all four loss terms, the CSV/savemat/PNG
writers, run_info.json, final_residual_by_frame.csv -- actually executes and
produces well-formed artifacts, BEFORE spending GPU time on the 5070.

    python3 run.py --root /path/to/project --epochs 4
"""
import argparse, os, runpy, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(HERE.parent.parent), help="project root holding train_inr_unsup_spiral.py")
ap.add_argument("--data", default=str(HERE / "synth_gassp.mat"))
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--out", default=str(HERE / "out"))
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
args = ap.parse_args()

sys.path.insert(0, str(HERE / "stub"))          # fake tinycudann wins over the real one
sys.path.insert(0, args.root)

import torch
print("=" * 72)
print("CPU DRY RUN -- NOT A SCIENTIFIC RESULT")
print("  substitution 1: tinycudann -> CPU stub (no hash encoding, no FullyFusedMLP)")
print("  substitution 2: INR.build_pos device 'cuda' -> cpu")
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print("=" * 72)

import model as _model
_orig = _model.INR.build_pos


def build_pos_cpu(self, grid_size, frame_num, time_coords=None):
    xs = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
    ys = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
    if time_coords is None:
        ts = torch.linspace(1 / (2 * frame_num), 1 - 1 / (2 * frame_num), frame_num)
    else:
        ts = torch.as_tensor(time_coords, dtype=torch.float32)
        if ts.numel() != frame_num:
            raise ValueError("time_coords length must match frame_num")
    xv, yv, tv = torch.meshgrid([xs, ys, ts], indexing="ij")
    return torch.stack((tv.flatten(), yv.flatten(), xv.flatten())).t()


_model.INR.build_pos = build_pos_cpu

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
os.chdir(out)                                    # log_path is './log/...' relative to CWD
sys.argv = ["train_inr_unsup_spiral.py",
            "--epochs", str(args.epochs),
            "--seed", "0",
            "--data_path", os.path.abspath(args.data),
            "--summary_epoch", "2",
            "--tag", "dryrun"] + args.extra
print("argv:", " ".join(sys.argv), "\ncwd:", os.getcwd(), "\n")

t0 = time.time()
runpy.run_path(os.path.join(args.root, "train_inr_unsup_spiral.py"), run_name="__main__")
print(f"\ndry run finished in {time.time() - t0:.1f} s")
