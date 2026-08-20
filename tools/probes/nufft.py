#!/usr/bin/env python3
"""
probes/nufft.py -- what does --kb_grid_size actually change?

Three DIFFERENT things get called "grid" in this pipeline; only one of them is
tied to the field of view:

  FOV         set by the acquisition (FOV = 1/dk).  No reconstruction
              parameter changes it.
  im_size     the reconstruction matrix, N = 216 here.  voxel = FOV/216.
              This is `N` from gassp1_data.mat, passed to SpiralNUFFT.
  grid_size   an INTERNAL oversampled Cartesian buffer used inside the
              Kaiser-Bessel gridding NUFFT.  torchkbnufft's own docstring:
              "Size of grid to use for interpolation, typically 1.25 to 2
              times im_size".  It never leaves the operator.  It has NO
              effect on FOV, voxel size, image position or scale -- only on
              the interpolation error and on peak memory.

`--kb_grid_size` sets the THIRD one.  Default None -> torchkbnufft picks
2 * im_size = 432.

This script proves both halves empirically, CPU only:
  (a) accuracy    -- forward-project against an exact NUDFT gold standard
  (b) geometry    -- a point source lands on the same pixel for every grid_size

    python3 probes/nufft.py
    python3 probes/nufft.py --N 216 --samples 3315 --coils 28
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import torch


def spiral_traj(samples: int, turns: int = 24) -> np.ndarray:
    """Archimedean single-arm spiral out to |k| = pi rad/voxel."""
    t = np.linspace(0.0, 1.0, samples)
    r, th = 0.5 * t, 2 * np.pi * turns * t
    return np.stack([r * np.cos(th), r * np.sin(th)])[None] * 2 * np.pi


def phantom(N: int) -> np.ndarray:
    ax = np.arange(N) - N / 2 + 0.5
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    img = (((X / 70) ** 2 + ((Y / 85) ** 2)) <= 1).astype(np.complex128)
    img[(X + 25) ** 2 + (Y - 8) ** 2 <= 20 ** 2] = 2.0
    return img * np.exp(1j * (0.004 * X - 0.003 * Y))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=216, help="im_size (the real matrix)")
    ap.add_argument("--samples", type=int, default=3315)
    ap.add_argument("--coils", type=int, default=28, help="only used for the memory column")
    ap.add_argument("--frames", type=int, default=50, help="only used for the memory column")
    ap.add_argument("--grids", type=int, nargs="+", default=[216, 270, 324, 432, 648])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()
    sys.path.insert(0, args.root)
    from spiral_nufft import SpiralNUFFT                            # noqa: E402

    N, S = args.N, args.samples
    ktraj_np = spiral_traj(S)
    img_np = phantom(N)
    n_shift = (N / 2, N / 2)
    print(f"im_size={N}  samples={S}  |k|max={np.abs(ktraj_np).max():.4f} rad "
          f"(pi={np.pi:.4f}; |k|=pi is the Nyquist edge of the {N}-matrix)")

    # ---- (a) accuracy vs an exact NUDFT (no gridding approximation at all)
    c = np.arange(N) - n_shift[0]
    Ex = np.exp(-1j * np.outer(ktraj_np[0, 0], c))
    Ey = np.exp(-1j * np.outer(ktraj_np[0, 1], np.arange(N) - n_shift[1]))
    y_exact = np.einsum("si,ij,sj->s", Ex, img_np, Ey, optimize=True)

    img_t = torch.as_tensor(img_np[None, None], dtype=torch.complex64)
    smap = torch.ones(1, N, N, dtype=torch.complex64)
    wi = torch.ones(1, S)
    ktraj = torch.as_tensor(ktraj_np, dtype=torch.float32)

    print(f"\n(a) forward accuracy vs exact NUDFT      "
          f"[gridded tensor = {args.frames}f x {args.coils}c x grid^2 complex64]")
    print(f"{'grid_size':>10} {'x im_size':>10} {'rel. error':>13} {'gridded tensor':>16}")
    out = {}
    for g in args.grids:
        op = SpiralNUFFT(ktraj, smap, wi, N, torch.device("cpu"),
                         kb_grid_size=g, n_shift=n_shift)
        with torch.no_grad():
            y = op.forward(img_t).numpy().ravel() * N     # SpiralNUFFT.forward divides by N
        out[g] = y
        err = np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact)
        mem = args.frames * args.coils * g * g * 8 / 2 ** 30
        print(f"{g:10d} {g / N:9.2f}x {err:13.3e} {mem:13.2f} GiB")
    if 324 in out and 432 in out:
        d = np.linalg.norm(out[324] - out[432]) / np.linalg.norm(out[432])
        print(f"\n    324 vs 432 differ from EACH OTHER by {d:.3e} "
              "-- compare that to the image-gate thresholds (5e-2 .. 1.5e-1).")

    # ---- (b) geometry: does grid_size move or rescale the image?
    print("\n(b) geometry: a point source, forward then adjoint")
    p = (70, 140)
    pt = torch.zeros(1, 1, N, N, dtype=torch.complex64)
    pt[0, 0, p[0], p[1]] = 1.0
    print(f"    true position (row,col) = {p}")
    print(f"{'grid_size':>10} {'peak (row,col)':>16} {'peak value':>12}")
    for g in args.grids:
        op = SpiralNUFFT(ktraj, smap, wi, N, torch.device("cpu"),
                         kb_grid_size=g, n_shift=n_shift)
        with torch.no_grad():
            rec = np.abs(op.adjoint(op.forward(pt), weighted=False)[0, 0].numpy())
        idx = np.unravel_index(rec.argmax(), rec.shape)
        print(f"{g:10d} {str(tuple(int(i) for i in idx)):>16} {rec.max():12.5g}")
    print("\n    Same pixel, same amplitude, every grid_size -> grid_size is NOT the FOV\n"
          "    and not the matrix.  It is an internal interpolation buffer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
