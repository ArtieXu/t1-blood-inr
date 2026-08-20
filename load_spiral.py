"""
load_spiral.py -- Stage 1 data adapter for GASSP1 spiral INR recon.

Reads gassp1_data.mat (from export_gassp1.m) and produces arrays for the INR
forward model:
    kdata  [phases, coils, samples]   complex
    ktraj  [phases, 2, samples]       float, radians in [-pi, pi]
    smap   [coils, N, N]              complex
    sinv   [coils, N, N]              complex   (pseudo-inverse maps, for adjoint)
    wi     [phases, samples]          float, density compensation

Run directly for the adjoint validation gate (torch-free numpy NUDFT):
    python3 load_spiral.py
Saves zero-filling adjoint images (density-weighted vs unweighted) plus the
authors' CG recon as a reference, and prints scaling stats. This is the gate
that must pass before the INR recon stage, and it settles the DC-weighting
choice (weighted vs plain).
"""
import os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, 'gassp1_data.mat')
# authors' recon (reference only); has CG recon + zero-filling for comparison
REF_MAT = '/ibic/projects3/ORCA/zechenxu/JHU/T1_Blood/20240926_ML/new_initial.mat'


def _read_complex(f, key):
    a = f[key][:]
    if a.dtype.names:                       # v7.3 complex = compound (real, imag)
        a = a['real'] + 1j * a['imag']
    return a


def calc_dcf_spiral(traj_sp, imsize):
    """Port of calc_dcf_Spiral.m. traj_sp [samples, phases] complex -> wi [samples, phases]."""
    npe = traj_sp.shape[1]
    r = np.abs(traj_sp[:, :1])              # radius profile of arm 0 (all arms identical, rotated)
    r = np.repeat(r, npe, axis=1)           # [samples, phases]
    rp1 = np.vstack([r[1:], r[-1:]])        # r([2:end, end])
    rm1 = np.vstack([r[:1], r[:-1]])        # r([1, 1:end-1])
    wi = rp1 - rm1
    wi[1:-1] = wi[1:-1] / 2
    wi = wi * 2 * np.pi * r / npe
    wi = wi * np.prod(imsize)
    return wi


def load_spiral_data(mat_path=MAT, apply_shift=True, swap_kxky=False):
    with h5py.File(mat_path, 'r') as f:
        # h5py reverses MATLAB axis order; comments give the h5 (python) shape
        k = _read_complex(f, 'k_data')      # (coils, phases, samples)
        traj = _read_complex(f, 'Traj')     # (phases, samples)
        S = _read_complex(f, 'S')           # (coils, col, row)  <- spatial axes swapped
        Sinv = _read_complex(f, 'Sinv')     # (coils, col, row)  <- spatial axes swapped
        mask = np.array(f['mask'][:], dtype=np.float32) if 'mask' in f else None
        N = int(np.array(f['N']).ravel()[0])
        shift = np.array(f['shift']).ravel()
    # h5py reverses ALL axes of v7.3 arrays, so the two spatial axes of the maps
    # arrive transposed vs the omega-defined (MATLAB) image frame. Restore them,
    # else the coil combine is mis-oriented (torch I0 != MATLAB I0; corr ~0.84).
    S = np.transpose(S, (0, 2, 1))          # -> (coils, row, col)
    Sinv = np.transpose(Sinv, (0, 2, 1))    # -> (coils, row, col)
    if mask is not None:
        mask = mask.T                       # -> (row, col)
    if mask is None:
        mask = np.ones((N, N), dtype=np.float32)

    wi = calc_dcf_spiral(traj.T, (N, N)).T  # -> (phases, samples)

    kx = np.real(traj)                      # (phases, samples)
    ky = np.imag(traj)
    if swap_kxky:
        kx, ky = ky, kx

    if apply_shift:                         # FOV-shift phase ramp (replicates @NUFFT shift([2,1]))
        sx, sy = shift[1], shift[0]
        ramp = np.exp(-1j * 2 * np.pi * (kx * sx + ky * sy))
        k = k * ramp[None, :, :]

    ktraj = np.stack([kx, ky], axis=1) * 2 * np.pi   # (phases, 2, samples) radians
    kdata = np.transpose(k, (1, 0, 2))               # (phases, coils, samples)
    return dict(kdata=kdata, ktraj=ktraj, smap=S, sinv=Sinv, wi=wi, mask=mask, N=N, shift=shift)


def adjoint_nudft(kdata, ktraj, sinv, wi=None, frames=None, n_shift=None):
    """
    Torch-free adjoint NUDFT, coil-combined zero-filling recon (validation only).
    kdata [P,C,S], ktraj [P,2,S] rad, sinv [C,N,N], wi [P,S] or None.
    Returns img [len(frames), N, N] complex.
    """
    P, C, S = kdata.shape
    N = sinv.shape[-1]
    frames = range(P) if frames is None else frames
    if n_shift is None:
        coords0 = np.arange(N) - N // 2
        coords1 = np.arange(N) - N // 2
    else:
        coords0 = np.arange(N) - n_shift[0]
        coords1 = np.arange(N) - n_shift[1]
    out = np.zeros((len(frames), N, N), dtype=np.complex128)
    for i, p in enumerate(frames):
        Ex = np.exp(1j * np.outer(ktraj[p, 0], coords0))  # [S,N], adjoint uses +i
        Ey = np.exp(1j * np.outer(ktraj[p, 1], coords1))  # [S,N]
        w = wi[p] if wi is not None else np.ones(S)
        img = np.zeros((N, N), dtype=np.complex128)
        for c in range(C):
            wd = w * kdata[p, c]
            img += sinv[c] * (Ex.T @ (wd[:, None] * Ey))
        out[i] = img
    return out


def _norm(x):
    x = np.abs(x)
    return x / (x.max() + 1e-12)


def _adjoint_gate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import io

    d = load_spiral_data(apply_shift=False)
    frames = [0, 25, 49]
    n_shift = (d['N'] / 2 + d['shift'][1], d['N'] / 2 + d['shift'][0])
    print('shapes: kdata', d['kdata'].shape, 'ktraj', d['ktraj'].shape,
          'smap', d['smap'].shape, 'wi', d['wi'].shape, 'N', d['N'])
    print('ktraj |k|max = %.3f rad (pi=%.3f)' % (np.abs(d['ktraj']).max(), np.pi))

    img_w = adjoint_nudft(d['kdata'], d['ktraj'], d['sinv'], wi=d['wi'], frames=frames, n_shift=n_shift)
    img_u = adjoint_nudft(d['kdata'], d['ktraj'], d['sinv'], wi=None, frames=frames, n_shift=n_shift)

    # reference: authors' CG recon (50 phases) for orientation/structure
    ref = io.loadmat(REF_MAT, variable_names=['I_GASSP_V_a1'])['I_GASSP_V_a1']  # (216,216,50)

    n = len(frames)
    fig, ax = plt.subplots(3, n, figsize=(4 * n, 12))
    for j, p in enumerate(frames):
        ax[0, j].imshow(_norm(img_u[j]), cmap='gray'); ax[0, j].set_title('unweighted adj  ph%d' % p)
        ax[1, j].imshow(_norm(img_w[j]), cmap='gray'); ax[1, j].set_title('wi-weighted adj  ph%d' % p)
        ax[2, j].imshow(_norm(ref[:, :, p]), cmap='gray'); ax[2, j].set_title('authors CG ref  ph%d' % p)
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout()
    out_png = os.path.join(HERE, 'adjoint_check.png')
    plt.savefig(out_png, dpi=90)
    print('saved', out_png)

    # scaling stat: center vs edge energy (why weighting may matter)
    for name, img in [('unweighted', img_u), ('weighted', img_w)]:
        a = np.abs(img[0])
        N = a.shape[0]
        cen = a[N//2-20:N//2+20, N//2-20:N//2+20].mean()
        edge = a[:20, :20].mean()
        print('%-11s center/edge mag ratio = %.2f' % (name, cen / (edge + 1e-12)))


if __name__ == '__main__':
    _adjoint_gate()
