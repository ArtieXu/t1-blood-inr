#!/usr/bin/env python3
"""
fake_data.py -- build a structurally faithful stand-in for gassp1_data.mat.

Same keys, same axis order, same MATLAB v7.3 complex layout that load_spiral.py
expects, but small and synthetic so the whole pipeline can be exercised on a CPU
in seconds. Values are meaningless; only shapes, dtypes and layout matter.

    python3 fake_data.py --out synth.mat --N 64 --frames 10 --coils 4 --samples 800
"""
import argparse
import h5py
import numpy as np

C64 = np.dtype([("real", "<f8"), ("imag", "<f8")])       # MATLAB v7.3 complex


def as_matlab_complex(a):
    out = np.empty(a.shape, dtype=C64)
    out["real"], out["imag"] = a.real, a.imag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_gassp.mat")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--coils", type=int, default=4)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--turns", type=int, default=8)
    a = ap.parse_args()
    N, F, C, S = a.N, a.frames, a.coils, a.samples
    rng = np.random.default_rng(0)

    # single-arm spiral, golden-angle rotated per frame, |k| <= 0.5 cycles/FOV
    t = np.linspace(0, 1, S)
    base = 0.5 * t * np.exp(1j * 2 * np.pi * a.turns * t)
    golden = np.deg2rad(137.51) * np.arange(F)
    traj = base[None, :] * np.exp(1j * golden)[:, None]          # (F, S)

    # smooth coil maps, never all-zero
    ax = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    smaps = []
    for c in range(C):
        ang = 2 * np.pi * c / C
        smaps.append(np.exp(-((X - 0.6 * np.cos(ang)) ** 2 + (Y - 0.6 * np.sin(ang)) ** 2) / 1.2)
                     * np.exp(1j * 0.3 * (X * np.cos(ang) + Y * np.sin(ang))))
    S_maps = np.stack(smaps)                                      # (C, N, N)
    rss = np.sqrt((np.abs(S_maps) ** 2).sum(0, keepdims=True)).clip(1e-3)
    S_maps = S_maps / rss
    Sinv = np.conj(S_maps) / (rss ** 2)

    kdata = (rng.standard_normal((C, F, S)) + 1j * rng.standard_normal((C, F, S))) * 1e2
    mask = ((X ** 2 + Y ** 2) <= 0.85 ** 2).astype(np.float32)

    with h5py.File(a.out, "w") as f:
        # load_spiral.py reads these exact names; h5py reverses MATLAB axis order,
        # so S/Sinv are written transposed the same way the real export produces.
        f.create_dataset("k_data", data=as_matlab_complex(kdata))          # (C,F,S)
        f.create_dataset("Traj",   data=as_matlab_complex(traj))           # (F,S)
        f.create_dataset("S",      data=as_matlab_complex(np.transpose(S_maps, (0, 2, 1))))
        f.create_dataset("Sinv",   data=as_matlab_complex(np.transpose(Sinv, (0, 2, 1))))
        f.create_dataset("mask",   data=mask.T.astype(np.float32))
        f.create_dataset("N",      data=np.array([[float(N)]]))
        f.create_dataset("shift",  data=np.array([[0.0], [0.0]]))
    print(f"wrote {a.out}: N={N} frames={F} coils={C} samples={S}")


if __name__ == "__main__":
    main()
