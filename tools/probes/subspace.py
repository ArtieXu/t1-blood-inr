#!/usr/bin/env python3
"""
probes/subspace.py -- read-only oracle-projection ceiling audit for the T1 prior.

Question: for a HARD rank-K temporal subspace Phi, what is the BEST any
reconstruction could do?  Answer = project the fully-sampled pre-B0 target
onto Phi and score it.  Nothing that uses Phi as a hard constraint can beat
this number, so a failing oracle projection means the PRIOR is the blocker,
not the optimizer.

CPU only, numpy/scipy only.  No GPU, no torch, no network.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- basis build
def build_ir_basis(ti_ms, t1_grid_ms, mz_grid, rank, t1_chunk=25):
    """Chunked Gram eigendecomposition; identical construction to
    run_coeff_subspace_exact_gate.ipynb cell 6 / MATLAB getT1Prior."""
    gram = np.zeros((ti_ms.size, ti_ms.size), dtype=np.float64)
    for s in range(0, t1_grid_ms.size, t1_chunk):
        t1 = t1_grid_ms[s:s + t1_chunk]
        curves = 1.0 + (mz_grid[None, None, :] - 1.0) * np.exp(-ti_ms[:, None, None] / t1[None, :, None])
        curves = curves.reshape(ti_ms.size, -1)
        curves /= np.linalg.norm(curves, axis=0, keepdims=True)
        gram += curves @ curves.T
    ev, evec = np.linalg.eigh(gram)
    order = np.argsort(ev)[::-1]
    ev = np.clip(ev[order], 0, None)
    basis = evec[:, order[:rank]]
    for c in range(rank):                       # deterministic sign
        p = np.argmax(np.abs(basis[:, c]))
        if basis[p, c] < 0:
            basis[:, c] *= -1
    return basis.astype(np.float64), ev


def data_svd_basis(X, rank, weights=None):
    """Oracle data-derived temporal basis: leading left singular vectors of the
    target itself.  Real basis (matches how Phi is applied to complex data)."""
    F = X.shape[0]
    A = X.reshape(F, -1)
    if weights is not None:
        A = A * weights.reshape(1, -1)
    G = (A @ A.conj().T).real
    ev, evec = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    return evec[:, order[:rank]].astype(np.float64), np.clip(ev[order], 0, None)


# ------------------------------------------------------------------- metrics
def project(X, basis):
    F = X.shape[0]
    flat = X.reshape(F, -1)
    return (basis @ (basis.T @ flat)).reshape(X.shape)


def nrmse(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def gradient_nrmse(pred, truth):
    p, t = np.abs(pred).squeeze(1), np.abs(truth).squeeze(1)
    err = sum(np.linalg.norm(np.diff(p, axis=ax) - np.diff(t, axis=ax)) ** 2 for ax in (1, 2))
    pw  = sum(np.linalg.norm(np.diff(t, axis=ax)) ** 2 for ax in (1, 2))
    return float(np.sqrt(err / pw))


def highpass_nrmse(pred, truth):
    p, t = np.abs(pred).squeeze(1), np.abs(truth).squeeze(1)
    ph = p - gaussian_filter(p, sigma=(0, 1, 1))
    th = t - gaussian_filter(t, sigma=(0, 1, 1))
    return float(np.linalg.norm(ph - th) / np.linalg.norm(th))


def shell_residual(pred, truth, n_shell=4):
    """Complex residual energy fraction by normalized Cartesian |k| shell.
    Reproduces the audit table in PROGRESS_AND_OUTLOOK_20260811.md section 4
    using the fully sampled grid instead of the spiral samples."""
    F, _, H, W = truth.shape
    K_t = np.fft.fftshift(np.fft.fft2(truth[:, 0], norm='ortho'), axes=(-2, -1))
    K_p = np.fft.fftshift(np.fft.fft2(pred[:, 0],  norm='ortho'), axes=(-2, -1))
    ky = (np.arange(H) - H // 2) / (H / 2)
    kx = (np.arange(W) - W // 2) / (W / 2)
    r = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    r = r / r[H // 2, :].max()                    # normalize by kx-max radius
    edges = np.linspace(0, 1, n_shell + 1)
    out = {}
    for i in range(n_shell):
        lo, hi = edges[i], edges[i + 1]
        m = (r >= lo) & (r < hi if i < n_shell - 1 else r <= 1.0)
        e = np.abs(K_p[:, m] - K_t[:, m]) ** 2
        pw = np.abs(K_t[:, m]) ** 2
        out[f'shell_{lo:.2f}_{hi:.2f}'] = float(np.sqrt(e.sum() / pw.sum()))
    return out


def score(pred, truth, label, extra=None):
    row = {
        'basis': label,
        'complex_NRMSE': nrmse(pred, truth),
        'magnitude_NRMSE': nrmse(np.abs(pred), np.abs(truth)),
        'gradient_NRMSE': gradient_nrmse(pred, truth),
        'highpass_NRMSE': highpass_nrmse(pred, truth),
    }
    row.update(shell_residual(pred, truth))
    # locked P2 gate (build_p2_retrospective_t1_ablation_notebook.py GATES):
    # magnitude<0.05, gradient<0.15, highpass<0.15.  uniform_DC<0.01,
    # outer-shell<0.05 and off_edge_ringing<0.10 need the spiral operator and
    # are NOT evaluated here.
    row['image_gate_pass'] = bool(row['magnitude_NRMSE'] < 0.05
                                  and row['gradient_NRMSE'] < 0.15
                                  and row['highpass_NRMSE'] < 0.15)
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='results/full_spiral_preb0_target.mat')
    ap.add_argument('--target_var', default='img_cg')
    ap.add_argument('--ti_start', type=float, default=50.0,
                    help='50 = retrospective reference contract; 63 = measured acquisition')
    ap.add_argument('--ti_step', type=float, default=200.0)
    ap.add_argument('--ranks', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 10])
    ap.add_argument('--support_quantile', type=float, default=None,
                    help='optional: zero voxels below this quantile of the time-max magnitude')
    ap.add_argument('--out', default='projection_audit')
    args = ap.parse_args()

    X = sio.loadmat(args.target)[args.target_var]
    if X.ndim == 3:                     # (H,W,F) -> (F,1,H,W)
        X = X.transpose(2, 0, 1)[:, None]
    X = X.astype(np.complex128)
    F = X.shape[0]
    if args.support_quantile is not None:
        m = np.abs(X).max(axis=0, keepdims=True)
        X = X * (m >= np.quantile(m, args.support_quantile))
    ti = args.ti_start + args.ti_step * np.arange(F, dtype=np.float64)
    print(f'target {args.target}:{args.target_var} shape={X.shape} '
          f'TI={ti[0]:.0f}..{ti[-1]:.0f} ms')

    T1_DEFAULT = np.linspace(200, 5000, 1500)
    MZ_DEFAULT = np.linspace(-1.0, 0.0, 1000)
    dicts = {
        'physical(T1 200-5000, Mz -1..0)':  (T1_DEFAULT, MZ_DEFAULT),
        'physical(T1 200-5000, Mz -1..+1)': (T1_DEFAULT, np.linspace(-1.0, 1.0, 1000)),
        'physical(T1 100-8000, Mz -1..0)':  (np.linspace(100, 8000, 1500), MZ_DEFAULT),
    }

    rows = []
    for name, (t1g, mzg) in dicts.items():
        for K in args.ranks:
            B, ev = build_ir_basis(ti, t1g, mzg, K)
            rows.append(score(project(X, B), X, f'{name} K={K}',
                              {'rank': K, 'family': name,
                               'dict_energy_retained': float(ev[:K].sum() / ev.sum())}))
            print(f"  {rows[-1]['basis']:44s} mag={rows[-1]['magnitude_NRMSE']:.4f} "
                  f"grad={rows[-1]['gradient_NRMSE']:.4f} hp={rows[-1]['highpass_NRMSE']:.4f} "
                  f"{'PASS' if rows[-1]['image_gate_pass'] else 'fail'}")

    for K in args.ranks:
        B, _ = data_svd_basis(X, K)
        rows.append(score(project(X, B), X, f'data-derived SVD K={K}',
                          {'rank': K, 'family': 'data-derived SVD'}))
        print(f"  {rows[-1]['basis']:44s} mag={rows[-1]['magnitude_NRMSE']:.4f} "
              f"grad={rows[-1]['gradient_NRMSE']:.4f} hp={rows[-1]['highpass_NRMSE']:.4f} "
              f"{'PASS' if rows[-1]['image_gate_pass'] else 'fail'}")

    # hybrid: physical rank-5 + r data-derived residual modes
    B5, _ = build_ir_basis(ti, T1_DEFAULT, MZ_DEFAULT, 5)
    R = X.reshape(F, -1) - B5 @ (B5.T @ X.reshape(F, -1))
    for r_extra in [1, 2, 3, 4, 5, 6]:
        Br, _ = data_svd_basis(R.reshape(X.shape), r_extra)
        Br = Br - B5 @ (B5.T @ Br)
        Q, _ = np.linalg.qr(np.concatenate([B5, Br], axis=1))
        rows.append(score(project(X, Q), X, f'physical K=5 + {r_extra} residual modes',
                          {'rank': 5 + r_extra, 'family': 'hybrid physical+residual'}))
        print(f"  {rows[-1]['basis']:44s} mag={rows[-1]['magnitude_NRMSE']:.4f} "
              f"grad={rows[-1]['gradient_NRMSE']:.4f} hp={rows[-1]['highpass_NRMSE']:.4f} "
              f"{'PASS' if rows[-1]['image_gate_pass'] else 'fail'}")

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(f'{args.out}.csv', index=False)
        print('wrote', f'{args.out}.csv')
    except ImportError:
        pass
    Path(f'{args.out}.json').write_text(json.dumps(
        {'target': args.target, 'ti_start': args.ti_start, 'frames': F, 'rows': rows}, indent=2))
    print('wrote', f'{args.out}.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
