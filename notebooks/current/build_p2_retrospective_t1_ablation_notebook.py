#!/usr/bin/env python3
"""Build the minimal P2 retrospective T1-prior ablation notebook."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "run_p2_retrospective_t1_ablation.ipynb"


def split_source(value: str) -> list[str]:
    return value.strip("\n").splitlines(keepends=True)


def make_cell(cell_type: str, value: str, index: int) -> dict:
    result = {
        "cell_type": cell_type,
        "id": hashlib.sha1(f"p2-retro-t1-ablation-v1-{index}".encode()).hexdigest()[:12],
        "metadata": {},
        "source": split_source(value),
    }
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


cells: list[dict] = []


def markdown(value: str) -> None:
    cells.append(make_cell("markdown", value, len(cells)))


def code(value: str) -> None:
    cells.append(make_cell("code", value, len(cells)))


markdown(
    r"""
# P2 retrospective ablation: T1 prior versus no-T1 under the same DCF

This notebook is the first controlled in-vivo retrospective gate. It uses the
real 50-frame fully sampled **pre-B0** reference as the known image series, but
it never uses measured GASSP1 k-space values. Instead, it forward-simulates the
fixed one-arm-per-TI GASSP1 encoding with the real GASSP1 trajectory, SMaps,
support, shift, and the same mean-normalized DCF.

The primary ablation is:

- `R5_T1_DCF`: rank-5 `X = Phi C` coefficient INR;
- `F0_noT1_DCF`: flexible 3D `(t,y,x)` image INR.

Both arms use seed 0, Adam with zero weight decay, LR `1e-3`, 800 uniform
parent updates, a fresh Adam optimizer, 1600 DCF continuation updates, and
checkpoint selection by the arm's own k-space objective only. Architectures
cannot be parameter-count matched because removing `Phi` changes a 2D
coefficient field into a 3D dynamic-image field.

Before either reconstruction, the notebook scores the zero-training-cost
`rank5_oracle_projection = Phi Phi^H X_ref`. This is the decisive model-bias
floor: if the oracle already loses real edges or high-frequency content, more
optimization cannot make the hard rank-5 model reproduce that component.

The fully sampled target is used only to synthesize k-space, set one global
amplitude scale, and score final saved outputs. It never selects a checkpoint.
This gate intentionally excludes measured k-space, noise, B0 forward modeling,
trajectory mismatch, motion, flow, wavelet, LLR, TV, and supervised loss.
"""
)

markdown("## 1. Mount Drive, lock inputs, and create a restart-safe result folder")

code(
    r"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import copy, hashlib, json, os, platform, random, shutil, subprocess, sys, time, zipfile
import numpy as np

EXPERIMENT_ID = '20260814_P2_fullref_T1_vs_noT1_DCF_v1'
PROTOCOL_REVISION = 'p2-retrospective-t1-ablation-v1'

DRIVE_ROOT = Path('/content/drive/MyDrive/T1_blood_INR_v2')
DRIVE_DATA = DRIVE_ROOT / '02_data_reference'
DRIVE_CODE = DRIVE_ROOT / '03_code'
DRIVE_RESULTS = DRIVE_ROOT / 'results' / 'retrospective_t1_ablation' / EXPERIMENT_ID
LOCAL = Path('/content') / f'T1_blood_INR_{EXPERIMENT_ID}'
SOURCE_LOCAL = LOCAL / 'source'

CODE_ZIP = DRIVE_CODE / 'T1_blood_INR_code_subspace_validation.zip'
RAW_PATH = DRIVE_DATA / 'gassp1_data.mat'
if not RAW_PATH.exists():
    RAW_PATH = DRIVE_ROOT / 'gassp1_data.mat'
FULLREF_PATH = DRIVE_DATA / 'full_spiral_preb0_target.mat'
SHARED_CURVE_PATH = DRIVE_DATA / 'full_spiral_shared_curve.mat'

EXPECTED_SHA256 = {
    'T1_blood_INR_code_subspace_validation.zip': 'ebefb30f3de81e1f82bcb2f6e734148826023508924a6a2536eafbf1be529ba6',
    'gassp1_data.mat': '59f6904f3ba3ea38301e40413f507fa8bb7b160c6f19ad96db1a5e9c1936232a',
    'full_spiral_preb0_target.mat': '442450e892592861fb799414cd4a68fcd81b3324bc22564046dcfcc97b8d56e5',
    'full_spiral_shared_curve.mat': '89019fc543ca611c22bcb1c63dca5ed0fe52129f510df3511beb86a7fca5cd7b',
}

required = [CODE_ZIP, RAW_PATH, FULLREF_PATH, SHARED_CURVE_PATH]
missing = [str(path) for path in required if not path.exists()]
assert not missing, 'Missing immutable Drive inputs:\n' + '\n'.join(missing)

LOCAL.mkdir(parents=True, exist_ok=True)
SOURCE_LOCAL.mkdir(parents=True, exist_ok=True)
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP) as archive:
    archive.extractall(LOCAL)
for path in [RAW_PATH, FULLREF_PATH, SHARED_CURVE_PATH]:
    shutil.copy2(path, SOURCE_LOCAL / path.name)
os.chdir(LOCAL)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + '.partial')
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)

def write_json_atomic(path, value):
    path = Path(path)
    local = LOCAL / f'.write_{path.name}'
    local.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False))
    copy_atomic(local, path)

input_paths = {
    'code_zip': CODE_ZIP,
    'raw_container': RAW_PATH,
    'fullref_target': FULLREF_PATH,
    'shared_curve': SHARED_CURVE_PATH,
}
input_hashes = {name: sha256(path) for name, path in input_paths.items()}
for name, path in input_paths.items():
    assert input_hashes[name] == EXPECTED_SHA256[path.name], f'Hash mismatch: {path}'

contract = {
    'format_version': 1,
    'protocol_revision': PROTOCOL_REVISION,
    'experiment_id': EXPERIMENT_ID,
    'question': 'Does a hard rank-5 T1 prior preserve and reconstruct real fully sampled dynamics under fixed one-arm-per-TI GASSP1 encoding and the same DCF?',
    'fixed': [
        'real fully sampled pre-B0 50-frame target clipped to the GASSP1 support',
        'synthetic k-space from the unweighted GASSP1 forward operator',
        'GASSP1 trajectory, SMaps, support, shift, 28 coils, and 3315 samples per TI',
        'mean-normalized analytic DCF used once in the squared complex residual',
        'seed 0, Adam weight_decay 0, lr 1e-3, 800 uniform plus 1600 DCF updates',
        'own k-space objective checkpoint selection; target never selects weights',
    ],
    'changed': 'hard rank-5 2D coefficient INR versus flexible 3D dynamic-image INR',
    'oracle': 'Phi Phi^H X_ref scored before training as the hard-prior model-bias floor',
    'reference_TI': 'loaded from full_spiral_shared_curve.mat; expected 50 + 200*n ms',
    'excluded': ['measured GASSP1 k-space values', 'noise', 'B0 forward model',
                 'trajectory mismatch', 'flow/motion model', 'TV', 'wavelet', 'LLR',
                 'target-selected checkpoint'],
    'interpretation_limit': 'retrospective sampling/prior/optimization gate; not a measured-k-space or B0-aware validation',
    'input_sha256': input_hashes,
}
contract_path = DRIVE_RESULTS / 'experiment_contract.json'
if contract_path.exists():
    assert json.loads(contract_path.read_text()) == contract, 'Existing experiment contract differs.'
else:
    write_json_atomic(contract_path, contract)

print('Output:', DRIVE_RESULTS)
print('Locked inputs:', input_hashes)
"""
)

markdown("## 2. Install the pinned T4 environment")

code(
    r"""
!nvidia-smi
!pip -q install torchkbnufft==1.5.2 ninja pandas scipy matplotlib pillow h5py tqdm
!pip -q install git+https://github.com/NVlabs/tiny-cuda-nn@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchkbnufft as tkbn
import tinycudann as tcnn
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from scipy import io
from scipy.ndimage import binary_dilation, gaussian_filter
from tqdm.auto import tqdm

assert torch.cuda.is_available(), 'A CUDA GPU is required.'
device = torch.device('cuda:0')
GPU_NAME = torch.cuda.get_device_name(0)
assert 'T4' in GPU_NAME, f'Use a Tesla T4 to match the locked phantom runs; found {GPU_NAME!r}.'

RUNTIME_SIGNATURE = {
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'torchkbnufft': getattr(tkbn, '__version__', 'unknown'),
    'tinycudann_commit': '749dd70c5afc5a9dadb85e5652ed65d55e0ba187',
    'gpu': GPU_NAME,
}
environment = {'python': sys.version, 'platform': platform.platform(), **RUNTIME_SIGNATURE}
write_json_atomic(DRIVE_RESULTS / 'environment.json', environment)
freeze = LOCAL / 'pip_freeze.txt'
freeze.write_text(subprocess.run(
    [sys.executable, '-m', 'pip', 'freeze'], check=True,
    capture_output=True, text=True).stdout)
copy_atomic(freeze, DRIVE_RESULTS / freeze.name)
print(environment)
"""
)

markdown("## 3. Load the real reference and synthesize the locked GASSP1 data")

code(
    r"""
from load_spiral import load_spiral_data
from spiral_nufft import SpiralNUFFT

spiral = load_spiral_data(SOURCE_LOCAL / 'gassp1_data.mat', apply_shift=False)
measured_kdata_shape = list(spiral['kdata'].shape)
del spiral['kdata']  # Measured values are forbidden in this retrospective gate.

N = int(spiral['N'])
support = np.asarray(spiral['mask'], dtype=bool)
support_bool = torch.as_tensor(support, dtype=torch.bool, device=device)
support32 = support_bool.to(torch.float32)
assert N == 216 and support.shape == (N, N)

target_file = io.loadmat(SOURCE_LOCAL / 'full_spiral_preb0_target.mat')
target_raw_np = np.asarray(target_file['img_cg'], dtype=np.complex64)
assert target_raw_np.shape == (50, 1, N, N), target_raw_np.shape
assert not bool(np.asarray(target_file['B0_deblurred']).squeeze())
target_raw_np = target_raw_np[:, 0]

raw_energy = np.linalg.norm(target_raw_np) ** 2
target_supported_np = target_raw_np * support[None]
support_retained_energy = float(np.linalg.norm(target_supported_np) ** 2 / raw_energy)
assert support_retained_energy > 0.85, support_retained_energy
target_scale = float(np.quantile(np.abs(target_supported_np)[:, support], 0.995))
assert np.isfinite(target_scale) and target_scale > 0
target_image_np = np.asarray(target_supported_np / target_scale, dtype=np.complex64)
target32 = torch.as_tensor(target_image_np, dtype=torch.complex64, device=device).unsqueeze(1)

operator32 = SpiralNUFFT(
    torch.as_tensor(spiral['ktraj']), torch.as_tensor(spiral['smap']),
    torch.as_tensor(spiral['wi']), N, device, sinv=None, support_mask=None,
    dcf_norm='mean', kb_grid_size=324,
    n_shift=(N / 2 + spiral['shift'][1], N / 2 + spiral['shift'][0]),
    numpoints=6,
)

def A_image32(image):
    return operator32.forward(image * support32)

def AH_image32(kspace):
    return operator32.adjoint(kspace, weighted=False) * support32

with torch.no_grad():
    y32 = A_image32(target32)

radius = np.sqrt(np.sum(np.asarray(spiral['ktraj'], dtype=np.float64) ** 2, axis=1))
radius /= radius.max()
SHELLS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
shell_names = [f'shell_{low:.2f}_{high:.2f}_DC' for low, high in SHELLS]
shell_masks_np = [(radius >= low) & (radius < high) for low, high in SHELLS]
shell_masks32 = [torch.as_tensor(mask, dtype=torch.float32, device=device)[:, None, :]
                 for mask in shell_masks_np]

wi_raw = np.asarray(spiral['wi'], dtype=np.float64)
wi_norm = wi_raw / wi_raw.mean()
assert np.isfinite(wi_norm).all() and wi_norm.min() > 0
assert np.allclose(wi_norm.mean(), 1.0, rtol=0, atol=1e-12)
assert np.allclose(radius, radius[:1], rtol=1e-7, atol=1e-9)
assert np.allclose(wi_norm, wi_norm[:1], rtol=1e-7, atol=1e-12)
w32 = torch.as_tensor(wi_norm, dtype=torch.float32, device=device)
assert float((w32 - operator32.wi).abs().max()) < 2e-6

target_power32 = y32.abs().square().sum().detach()
dcf_target_power32 = (w32[:, None, :] * y32.abs().square()).sum().detach()
shell_target_powers32 = [(y32.abs().square() * mask).sum().detach().clamp_min(1e-20)
                         for mask in shell_masks32]

def adjoint_stats32(x, y):
    ax = A_image32(x)
    ahy = AH_image32(y)
    left = torch.vdot(ax.reshape(-1), y.reshape(-1))
    right = torch.vdot(x.reshape(-1), ahy.reshape(-1))
    absolute = (left - right).abs()
    return {
        'inner_product_relative': float(
            absolute / torch.maximum(left.abs(), right.abs()).clamp_min(1e-20)),
        'norm_bound_relative': float(
            absolute / (torch.linalg.vector_norm(ax) * torch.linalg.vector_norm(y)).clamp_min(1e-20)),
    }

generator = torch.Generator(device=device).manual_seed(64200)
adjoint_checks = []
with torch.no_grad():
    for _ in range(3):
        x_probe = (torch.randn(target32.shape, generator=generator, device=device) +
                   1j * torch.randn(target32.shape, generator=generator, device=device)) * support32
        y_probe = (torch.randn(y32.shape, generator=generator, device=device) +
                   1j * torch.randn(y32.shape, generator=generator, device=device))
        adjoint_checks.append(adjoint_stats32(x_probe, y_probe))
assert max(item['norm_bound_relative'] for item in adjoint_checks) < 1e-5, adjoint_checks

dcf_rows = [{
    'statistic': 'global', 'min': wi_norm.min(), 'q25': np.quantile(wi_norm, .25),
    'median': np.median(wi_norm), 'q75': np.quantile(wi_norm, .75),
    'max': wi_norm.max(), 'mean': wi_norm.mean(),
}]
for (low, high), mask in zip(SHELLS, shell_masks_np):
    values = wi_norm[mask]
    dcf_rows.append({'statistic': f'shell_{low:.2f}_{high:.2f}',
                     'min': values.min(), 'q25': np.quantile(values, .25),
                     'median': np.median(values), 'q75': np.quantile(values, .75),
                     'max': values.max(), 'mean': values.mean()})
dcf_table = pd.DataFrame(dcf_rows)
dcf_table.to_csv(DRIVE_RESULTS / 'dcf_audit.csv', index=False)
np.save(DRIVE_RESULTS / 'dcf_mean_normalized.npy', wi_norm)

target_audit = {
    'source_shape': list(target_file['img_cg'].shape),
    'normalized_shape': list(target_image_np.shape),
    'source_B0_deblurred': False,
    'GASSP1_support_pixels': int(support.sum()),
    'support_retained_energy_fraction': support_retained_energy,
    'normalization': 'divide by target 99.5th magnitude percentile inside GASSP1 support',
    'target_scale': target_scale,
    'measured_kdata_shape_not_used': measured_kdata_shape,
    'synthetic_kdata_shape': list(y32.shape),
    'adjoint_checks': adjoint_checks,
}
write_json_atomic(DRIVE_RESULTS / 'target_operator_audit.json', target_audit)

rms = np.sqrt(np.mean(np.abs(target_raw_np) ** 2, axis=0))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
axes[0].imshow(rms, cmap='gray', vmax=np.quantile(rms, .995)); axes[0].set_title('full reference RMS')
axes[1].imshow(support, cmap='gray'); axes[1].set_title('GASSP1 support')
axes[2].imshow(rms, cmap='gray', vmax=np.quantile(rms, .995))
axes[2].contour(support, [.5], colors='r'); axes[2].set_title('locked support overlay')
for axis in axes: axis.axis('off')
fig.savefig(DRIVE_RESULTS / 'target_support_audit.png', dpi=160, bbox_inches='tight')
plt.show()

display(dcf_table)
print('Support retained energy:', support_retained_energy)
print('Adjoint norm-bound errors:', [item['norm_bound_relative'] for item in adjoint_checks])
"""
)

markdown("## 4. Build the reference-TI rank-5 basis and its oracle projection")

code(
    r"""
shared = io.loadmat(SOURCE_LOCAL / 'full_spiral_shared_curve.mat')
TI_MS = np.asarray(shared['TI'], dtype=np.float64).reshape(-1)
assert TI_MS.shape == (50,)
assert np.array_equal(TI_MS, 50 + 200 * np.arange(50)), TI_MS
shared_mask128 = np.asarray(shared['mask_shared'], dtype=bool)
assert shared_mask128.shape == (128, 128)
shared_mask16 = shared_mask128.reshape(16, 8, 16, 8).mean(axis=(1, 3)) >= 0.5
assert shared_mask16.sum() >= 10
shared_loc_matlab = np.asarray(shared['loc'], dtype=int).reshape(2)
roi_start = shared_loc_matlab - 9  # MATLAB loc + floor(-7.5:7.5), converted to Python.

def shared_ijv_curve(image_np):
    row, column = roi_start
    crop = image_np[:, row:row + 16, column:column + 16]
    assert crop.shape == (50, 16, 16)
    return np.abs(crop[:, shared_mask16].mean(axis=1))

target_ijv_curve = shared_ijv_curve(target_image_np)
assert np.linalg.norm(target_ijv_curve) > 0
T1_GRID_MS = np.linspace(200, 5000, 1500, dtype=np.float64)
MZ0_OVER_M0 = np.linspace(-1.0, 0.0, 1000, dtype=np.float64)
RANK = 5

def build_ir_basis(ti_ms, t1_grid_ms, mz_grid, rank, t1_chunk=25):
    gram = np.zeros((ti_ms.size, ti_ms.size), dtype=np.float64)
    for start in range(0, t1_grid_ms.size, t1_chunk):
        t1 = t1_grid_ms[start:start + t1_chunk]
        curves = 1.0 + (mz_grid[None, None, :] - 1.0) * np.exp(
            -ti_ms[:, None, None] / t1[None, :, None])
        curves = curves.reshape(ti_ms.size, -1)
        curves /= np.linalg.norm(curves, axis=0, keepdims=True)
        gram += curves @ curves.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    basis = eigenvectors[:, order[:rank]]
    for column in range(rank):
        pivot = np.argmax(np.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    retained = float(eigenvalues[:rank].sum() / eigenvalues.sum())
    return basis.astype(np.float32), eigenvalues, retained

basis, _, retained_energy = build_ir_basis(
    TI_MS, T1_GRID_MS, MZ0_OVER_M0, RANK)
assert np.allclose(basis.T @ basis, np.eye(RANK), atol=1e-5)
assert retained_energy > 0.99999

basis_path = LOCAL / 'fullref_ir_rank5_ti50_mzminus1to0.npy'
np.save(basis_path, basis)
copy_atomic(basis_path, DRIVE_RESULTS / basis_path.name)
basis32 = torch.as_tensor(basis, dtype=torch.complex64, device=device)

oracle_coeff_np = np.einsum('tk,thw->khw', basis, target_image_np)
oracle_image_np = np.einsum('tk,khw->thw', basis, oracle_coeff_np).astype(np.complex64)
def render_coeff32(coeff):
    return torch.einsum('tk,khw->thw', basis32, coeff * support32).unsqueeze(1)

basis_manifest = {
    'construction': 'signed IR curves -> per-curve L2 normalization -> leading left singular subspace',
    'reference_TI_ms': TI_MS.tolist(),
    'T1_grid_ms': [200, 5000, 1500],
    'Mz0_over_M0': [-1.0, 0.0, 1000],
    'rank': RANK,
    'retained_dictionary_energy': retained_energy,
    'basis_sha256': sha256(basis_path),
}
write_json_atomic(DRIVE_RESULTS / 'basis_manifest.json', basis_manifest)

fig, axis = plt.subplots(figsize=(8, 3), constrained_layout=True)
for column in range(RANK):
    axis.plot(TI_MS, basis[:, column], label=f'B{column + 1}')
axis.set_xlabel('reference TI (ms)')
axis.set_title('Reference-specific rank-5 T1 basis')
axis.grid(True); axis.legend(ncol=5)
fig.savefig(DRIVE_RESULTS / 'rank5_basis.png', dpi=150, bbox_inches='tight')
plt.show()
display(pd.Series(basis_manifest, name='basis contract'))
"""
)

markdown("## 5. Lock image gates and score the rank-5 oracle before training")

code(
    r"""
GATES = {
    'uniform_DC': 0.01,
    'magnitude_NRMSE': 0.05,
    'gradient_NRMSE': 0.15,
    'highpass_NRMSE': 0.15,
    'outer_shell_DC': 0.05,
    'off_edge_ringing_secondary': 0.10,
}

target_magnitude = np.abs(target_image_np)
gy = np.diff(target_magnitude, axis=1, append=target_magnitude[:, -1:, :])
gx = np.diff(target_magnitude, axis=2, append=target_magnitude[:, :, -1:])
spatial_edge_energy = np.sqrt(np.sum(gx ** 2 + gy ** 2, axis=0))
EDGE_THRESHOLD_FRACTION = 0.10
EDGE_DILATION_PIXELS = 4
edge_core = spatial_edge_energy >= EDGE_THRESHOLD_FRACTION * spatial_edge_energy.max()
edge_band = binary_dilation(edge_core, iterations=EDGE_DILATION_PIXELS)
flat_mask = support & ~edge_band
assert flat_mask.sum() > 1000

target_high = target_magnitude - gaussian_filter(target_magnitude, sigma=(0, 1, 1))
target_high_norm = np.linalg.norm(target_high[:, support])
assert target_high_norm > 0

def gradient_nrmse(prediction, target):
    pred, truth = np.abs(prediction), np.abs(target)
    error = sum(np.linalg.norm(np.diff(pred, axis=axis) - np.diff(truth, axis=axis)) ** 2
                for axis in (1, 2))
    power = sum(np.linalg.norm(np.diff(truth, axis=axis)) ** 2 for axis in (1, 2))
    return float(np.sqrt(error / power))

def highpass_nrmse(prediction, target):
    pred, truth = np.abs(prediction), np.abs(target)
    pred_high = pred - gaussian_filter(pred, sigma=(0, 1, 1))
    truth_high = truth - gaussian_filter(truth, sigma=(0, 1, 1))
    return float(np.linalg.norm((pred_high - truth_high)[:, support]) /
                 np.linalg.norm(truth_high[:, support]))

def off_edge_ringing(prediction):
    pred_mag = np.abs(prediction)
    pred_high = pred_mag - gaussian_filter(pred_mag, sigma=(0, 1, 1))
    return float(np.linalg.norm((pred_high - target_high)[:, flat_mask]) / target_high_norm)

def kspace_metrics32(prediction):
    error = (prediction - y32).abs().square()
    uniform_dc = error.sum() / target_power32
    dcf_dc = (w32[:, None, :] * error).sum() / dcf_target_power32
    shell_dc = [(error * mask).sum() / power
                for mask, power in zip(shell_masks32, shell_target_powers32)]
    return uniform_dc, dcf_dc, shell_dc

def score_image32(name, image_np):
    image = torch.as_tensor(image_np, dtype=torch.complex64, device=device).unsqueeze(1) * support32
    with torch.no_grad():
        prediction = A_image32(image)
        uniform_dc, dcf_dc, shell_dc = kspace_metrics32(prediction)
    image_np = image[:, 0].cpu().numpy()
    projection = np.einsum('tk,thw->khw', basis, image_np)
    projected_image = np.einsum('tk,khw->thw', basis, projection)
    row = {
        'arm': name,
        'uniform_DC': float(uniform_dc),
        'data_NRMSE': float(torch.sqrt(uniform_dc)),
        'DCF_DC': float(dcf_dc),
        'magnitude_NRMSE': float(np.linalg.norm(np.abs(image_np) - target_magnitude) /
                                 np.linalg.norm(target_magnitude)),
        'complex_NRMSE': float(np.linalg.norm(image_np - target_image_np) /
                               np.linalg.norm(target_image_np)),
        'gradient_NRMSE': gradient_nrmse(image_np, target_image_np),
        'highpass_NRMSE': highpass_nrmse(image_np, target_image_np),
        'off_edge_ringing': off_edge_ringing(image_np),
        'IJV_curve_NRMSE': float(
            np.linalg.norm(shared_ijv_curve(image_np) - target_ijv_curve) /
            np.linalg.norm(target_ijv_curve)),
        'out_of_T1_subspace_fraction': float(np.linalg.norm(image_np - projected_image) /
                                             np.linalg.norm(image_np)),
        'projected_coefficient_NRMSE_vs_oracle': float(
            np.linalg.norm(projection - oracle_coeff_np) / np.linalg.norm(oracle_coeff_np)),
    }
    row.update({key: float(value) for key, value in zip(shell_names, shell_dc)})
    return row, image_np

def common_image_gate(row):
    return bool(row['uniform_DC'] < GATES['uniform_DC']
                and row['magnitude_NRMSE'] < GATES['magnitude_NRMSE']
                and row['gradient_NRMSE'] < GATES['gradient_NRMSE']
                and row['highpass_NRMSE'] < GATES['highpass_NRMSE']
                and row['shell_0.75_1.01_DC'] < GATES['outer_shell_DC']
                and row['off_edge_ringing'] < GATES['off_edge_ringing_secondary'])

oracle_row, oracle_image_np = score_image32('rank5_oracle_projection', oracle_image_np)
oracle_row['common_image_gate_pass'] = common_image_gate(oracle_row)
pd.DataFrame([oracle_row]).to_csv(DRIVE_RESULTS / 'rank5_oracle_summary.csv', index=False)
display(pd.Series(oracle_row, name='rank-5 oracle'))

mask_fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
axes[0].imshow(spatial_edge_energy, cmap='magma'); axes[0].set_title('target edge energy')
axes[1].imshow(edge_band, cmap='gray'); axes[1].set_title('edge band: excluded')
axes[2].imshow(flat_mask, cmap='gray'); axes[2].set_title('flat region: ringing test')
for axis in axes: axis.axis('off')
mask_fig.savefig(DRIVE_RESULTS / 'ringing_mask_definition.png', dpi=160, bbox_inches='tight')
plt.show()
print('Locked gates:', GATES)
print('Oracle pass:', oracle_row['common_image_gate_pass'])
"""
)

markdown("## 6. Define the two models and one shared restart-safe training loop")

code(
    r"""
SEED = 0
LR = 1e-3
UNIFORM_PARENT_STEPS = 800
DCF_CONTINUATION_STEPS = 1600
OBJECTIVE_CHECK_EVERY = 20
RESUME_SAVE_EVERY = 100

MODEL_CONFIGS = {
    'R5_T1': {
        'family': 'rank5_coefficient', 'n_input_dims': 2,
        'n_levels': 16, 'n_features_per_level': 2, 'log2_hashmap_size': 20,
        'base_resolution': 16, 'finest_resolution': N,
        'n_neurons': 128, 'n_hidden_layers': 3, 'n_output_dims': 2 * RANK,
        'uniform_parent_schedule': 'all levels active',
    },
    'F0_noT1': {
        'family': 'flexible_3d_image', 'n_input_dims': 3,
        'n_levels': 16, 'n_features_per_level': 2, 'log2_hashmap_size': 24,
        'base_resolution': 16, 'finest_resolution': N,
        'n_neurons': 128, 'n_hidden_layers': 5, 'n_output_dims': 2,
        'time_coords': 'reference TI span mapped to [0.01, 0.99]',
        'uniform_parent_schedule': '4 levels to all 16 levels by step 500',
    },
}

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

class CoefficientINR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = MODEL_CONFIGS['R5_T1']
        scale = (N / config['base_resolution']) ** (1 / (config['n_levels'] - 1))
        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                'otype': 'HashGrid', 'n_levels': config['n_levels'],
                'n_features_per_level': config['n_features_per_level'],
                'log2_hashmap_size': config['log2_hashmap_size'],
                'base_resolution': config['base_resolution'],
                'per_level_scale': float(scale),
            })
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims, n_output_dims=2 * RANK,
            network_config={
                'otype': 'FullyFusedMLP', 'activation': 'ReLU',
                'output_activation': 'None', 'n_neurons': config['n_neurons'],
                'n_hidden_layers': config['n_hidden_layers'],
            })

    def forward(self, coordinates):
        values = self.network(self.encoding(coordinates)).float()
        values = values.reshape(N, N, RANK, 2)
        return torch.view_as_complex(values.contiguous()).permute(2, 0, 1)

class FlexibleINR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = MODEL_CONFIGS['F0_noT1']
        scale = (N / config['base_resolution']) ** (1 / (config['n_levels'] - 1))
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid', 'n_levels': config['n_levels'],
                'n_features_per_level': config['n_features_per_level'],
                'log2_hashmap_size': config['log2_hashmap_size'],
                'base_resolution': config['base_resolution'],
                'per_level_scale': float(scale),
            })
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims, n_output_dims=2,
            network_config={
                'otype': 'FullyFusedMLP', 'activation': 'ReLU',
                'output_activation': 'None', 'n_neurons': config['n_neurons'],
                'n_hidden_layers': config['n_hidden_layers'],
            })

    def forward(self, coordinates, active_levels=None):
        encoded = self.encoding(coordinates)
        if active_levels is not None and active_levels < 16:
            mask = torch.zeros_like(encoded)
            mask[:, :active_levels * 2] = 1
            encoded = encoded * mask
        return self.network(encoded).float()

axis = torch.linspace(1 / (2 * N), 1 - 1 / (2 * N), N, device=device)
coord_y, coord_x = torch.meshgrid(axis, axis, indexing='ij')
coordinates2d = torch.stack([coord_y.reshape(-1), coord_x.reshape(-1)], dim=1)
time_axis = torch.as_tensor(
    0.01 + 0.98 * (TI_MS - TI_MS.min()) / (TI_MS.max() - TI_MS.min()),
    dtype=torch.float32, device=device)
coord_y3, coord_x3, coord_t3 = torch.meshgrid(axis, axis, time_axis, indexing='ij')
coordinates3d = torch.stack(
    [coord_t3.reshape(-1), coord_y3.reshape(-1), coord_x3.reshape(-1)], dim=1)

def new_model(family):
    if family == 'R5_T1':
        return CoefficientINR().to(device)
    if family == 'F0_noT1':
        return FlexibleINR().to(device)
    raise ValueError(family)

def parent_active_levels(step):
    if step >= 500:
        return 16
    return min(16, 4 + int((step / 500) * 12))

def render_model(model, family, active_levels=None):
    if family == 'R5_T1':
        return render_coeff32(model(coordinates2d)) * support32
    values = model(coordinates3d, active_levels=active_levels)
    image = torch.view_as_complex(values.reshape(N, N, 50, 2).contiguous())
    return image.permute(2, 0, 1).unsqueeze(1) * support32

def image_objectives(image):
    prediction = A_image32(image)
    error = (prediction - y32).abs().square()
    uniform = error.sum() / target_power32
    dcf = (w32[:, None, :] * error).sum() / dcf_target_power32
    shells = [(error * mask).sum() / power
              for mask, power in zip(shell_masks32, shell_target_powers32)]
    return {'uniform': uniform, 'dcf': dcf,
            'shell': torch.stack(shells).mean(), 'outer_shell': shells[-1]}

@torch.no_grad()
def evaluate_model(model, family):
    image = render_model(model, family, active_levels=None)
    return image, image_objectives(image)

def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def optimizer_to_cpu(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for item in state.get('state', {}).values():
        for key, value in list(item.items()):
            if torch.is_tensor(value):
                item[key] = value.detach().cpu()
    return state

def stage_paths(stage):
    return (DRIVE_RESULTS / f'{stage}_resume.pt',
            DRIVE_RESULTS / f'{stage}_resume.pt.sha256')

def save_stage(stage, payload):
    path, sidecar = stage_paths(stage)
    local = LOCAL / f'{stage}_resume.pt'
    torch.save(payload, local)
    digest = sha256(local)
    copy_atomic(local, path)
    local_sha = LOCAL / f'{stage}_resume.pt.sha256'
    local_sha.write_text(digest)
    copy_atomic(local_sha, sidecar)

def load_stage(stage, family, objective, target_steps):
    path, sidecar = stage_paths(stage)
    if not path.exists() or not sidecar.exists() or sha256(path) != sidecar.read_text().strip():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    assert payload['experiment_id'] == EXPERIMENT_ID
    assert payload['runtime_signature'] == RUNTIME_SIGNATURE
    assert payload['family'] == family and payload['stage'] == stage
    assert payload['objective'] == objective and payload['target_steps'] == target_steps
    assert payload['model_config'] == MODEL_CONFIGS[family]
    return payload

def train_stage(family, stage, objective, target_steps, initial_state, use_ctf):
    reset_seed()
    model = new_model(family)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # weight_decay=0 for both arms
    payload = load_stage(stage, family, objective, target_steps)
    if payload is None:
        start_step = 0
        history, checks = [], []
        with torch.no_grad():
            _, values0 = evaluate_model(model, family)
        best_step, best_objective = 0, float(values0[objective])
        best_state = cpu_state(model)
        checks.append({'step': 0, 'objective': objective,
                       **{key: float(value) for key, value in values0.items()},
                       'is_new_best': True})
    else:
        start_step = int(payload['completed_step'])
        model.load_state_dict(payload['model_state'])
        optimizer.load_state_dict(payload['optimizer_state'])
        best_step = int(payload['best_step'])
        best_objective = float(payload['best_objective'])
        best_state = payload['best_model_state']
        history, checks = payload['history'], payload['checks']
        torch.set_rng_state(payload['torch_cpu_rng_state'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in payload['torch_cuda_rng_states']])
        np.random.set_state(payload['numpy_rng_state'])
        random.setstate(payload['python_rng_state'])
        print(f'Resumed {stage} at step {start_step}')

    start_time = time.time()
    loop = tqdm(range(start_step + 1, target_steps + 1), desc=f'{stage} from {start_step}')
    for step in loop:
        active = parent_active_levels(step) if use_ctf else None
        image = render_model(model, family, active_levels=active)
        values = image_objectives(image)
        loss = values[objective]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({'step': step, 'objective': objective,
                        'loss_before_update': float(loss.detach()),
                        **{f'{key}_before_update': float(value.detach())
                           for key, value in values.items()},
                        'active_levels': 16 if active is None else active})

        if step % OBJECTIVE_CHECK_EVERY == 0 or step == target_steps:
            with torch.no_grad():
                _, after = evaluate_model(model, family)
            own = float(after[objective])
            is_best = own < best_objective
            if is_best:
                best_step, best_objective = step, own
                best_state = cpu_state(model)
            checks.append({'step': step, 'objective': objective,
                           **{key: float(value) for key, value in after.items()},
                           'is_new_best': bool(is_best)})
            loop.set_postfix(own=f'{own:.2e}', best=f'{best_objective:.2e}')

        if step % RESUME_SAVE_EVERY == 0 or step == target_steps:
            save_stage(stage, {
                'experiment_id': EXPERIMENT_ID, 'runtime_signature': RUNTIME_SIGNATURE,
                'family': family, 'model_config': MODEL_CONFIGS[family],
                'stage': stage, 'objective': objective,
                'completed_step': step, 'target_steps': target_steps,
                'model_state': cpu_state(model),
                'optimizer_state': optimizer_to_cpu(optimizer),
                'best_model_state': best_state, 'best_step': best_step,
                'best_objective': best_objective,
                'history': history, 'checks': checks,
                'torch_cpu_rng_state': torch.get_rng_state().cpu(),
                'torch_cuda_rng_states': [value.cpu() for value in torch.cuda.get_rng_state_all()],
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            })
            for suffix, rows in [('history.csv', history), ('checkpoint_objectives.csv', checks)]:
                local_csv = LOCAL / f'{stage}_{suffix}'
                pd.DataFrame(rows).to_csv(local_csv, index=False)
                copy_atomic(local_csv, DRIVE_RESULTS / local_csv.name)

    final_state = cpu_state(model)
    best_model = new_model(family)
    best_model.load_state_dict(best_state)
    with torch.no_grad():
        best_image, best_values = evaluate_model(best_model, family)
        final_image, final_values = evaluate_model(model, family)
    metadata = {
        'family': family, 'stage': stage, 'objective': objective,
        'steps': target_steps, 'best_step': best_step,
        'best_objective': best_objective,
        'best_values': {key: float(value) for key, value in best_values.items()},
        'final_values': {key: float(value) for key, value in final_values.items()},
        'runtime_minutes_this_session': (time.time() - start_time) / 60,
        'checkpoint_selection': 'own k-space objective only; target never used',
        'optimizer': {'name': 'Adam', 'lr': LR, 'weight_decay': 0.0},
        'objective_check_every': OBJECTIVE_CHECK_EVERY,
        'resume_save_every': RESUME_SAVE_EVERY,
    }
    write_json_atomic(DRIVE_RESULTS / f'{stage}_meta.json', metadata)
    del model, best_model, optimizer
    torch.cuda.empty_cache()
    return (final_state, best_state, best_image[:, 0].cpu().numpy(),
            final_image[:, 0].cpu().numpy(), metadata)

def run_two_stage(family):
    parent_stage = f'{family}_P0_uniform'
    dcf_stage = f'{family}_DCF'
    parent_final_path = DRIVE_RESULTS / f'{parent_stage}_final.pt'
    if parent_final_path.exists():
        parent_final_state = torch.load(parent_final_path, map_location='cpu', weights_only=True)
        print('Loaded completed parent:', parent_stage)
    else:
        parent_final_state, _, _, _, _ = train_stage(
            family, parent_stage, 'uniform', UNIFORM_PARENT_STEPS,
            initial_state=None, use_ctf=(family == 'F0_noT1'))
        local_parent = LOCAL / parent_final_path.name
        torch.save(parent_final_state, local_parent)
        copy_atomic(local_parent, parent_final_path)

    complete_path = DRIVE_RESULTS / f'{dcf_stage}_complete.json'
    images_path = DRIVE_RESULTS / f'{dcf_stage}_images.mat'
    if complete_path.exists() and images_path.exists():
        payload = json.loads(complete_path.read_text())
        images = io.loadmat(images_path)
        best_image_np = np.asarray(images['best_image'], dtype=np.complex64)
        metadata = payload['metadata']
        assert payload['complete'] and metadata['family'] == family
        assert best_image_np.shape == (50, N, N), best_image_np.shape
        print('Loaded completed DCF stage:', dcf_stage)
    else:
        _, best_state, best_image_np, final_image_np, metadata = train_stage(
            family, dcf_stage, 'dcf', DCF_CONTINUATION_STEPS,
            initial_state=parent_final_state, use_ctf=False)
        local_best = LOCAL / f'{dcf_stage}_best.pt'
        torch.save(best_state, local_best)
        copy_atomic(local_best, DRIVE_RESULTS / local_best.name)
        local_mat = LOCAL / images_path.name
        io.savemat(local_mat, {'best_image': best_image_np, 'final_image': final_image_np})
        copy_atomic(local_mat, images_path)
        write_json_atomic(complete_path, {'metadata': metadata, 'complete': True})
    return best_image_np, metadata

parameter_counts = {}
for family in MODEL_CONFIGS:
    probe = new_model(family)
    parameter_counts[family] = int(sum(parameter.numel() for parameter in probe.parameters()))
    del probe
    torch.cuda.empty_cache()
print('Parameter counts:', parameter_counts)
"""
)

markdown("## 7. Run rank-5 T1+DCF")

code(
    r"""
t1_best_np, t1_meta = run_two_stage('R5_T1')
print(t1_meta)
"""
)

markdown("## 8. Run flexible no-T1+DCF")

code(
    r"""
f0_best_np, f0_meta = run_two_stage('F0_noT1')
print(f0_meta)
"""
)

markdown("## 9. Locked comparison and decision")

code(
    r"""
oracle_row, oracle_image_np = score_image32('rank5_oracle_projection', oracle_image_np)
oracle_row['common_image_gate_pass'] = common_image_gate(oracle_row)
t1_row, t1_image_np = score_image32('R5_T1_DCF', t1_best_np)
t1_row['common_image_gate_pass'] = common_image_gate(t1_row)
f0_row, f0_image_np = score_image32('F0_noT1_DCF', f0_best_np)
f0_row['common_image_gate_pass'] = common_image_gate(f0_row)

comparison = pd.DataFrame([oracle_row, t1_row, f0_row]).set_index('arm')
summary_local = LOCAL / 'P2_retrospective_summary.csv'
comparison.to_csv(summary_local)
copy_atomic(summary_local, DRIVE_RESULTS / summary_local.name)
display(comparison)

per_frame_rows = []
for name, image_np in [('rank5_oracle_projection', oracle_image_np),
                       ('R5_T1_DCF', t1_image_np), ('F0_noT1_DCF', f0_image_np)]:
    for frame, ti_ms in enumerate(TI_MS):
        truth = target_image_np[frame]
        estimate = image_np[frame]
        per_frame_rows.append({
            'arm': name, 'frame': frame, 'TI_ms': float(ti_ms),
            'magnitude_NRMSE': float(np.linalg.norm(np.abs(estimate) - np.abs(truth)) /
                                     np.linalg.norm(np.abs(truth))),
            'complex_NRMSE': float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
        })
per_frame = pd.DataFrame(per_frame_rows)
per_frame_local = LOCAL / 'P2_per_frame_metrics.csv'
per_frame.to_csv(per_frame_local, index=False)
copy_atomic(per_frame_local, DRIVE_RESULTS / per_frame_local.name)

curve_rows = []
for name, image_np in [('target', target_image_np),
                       ('rank5_oracle_projection', oracle_image_np),
                       ('R5_T1_DCF', t1_image_np), ('F0_noT1_DCF', f0_image_np)]:
    for frame, (ti_ms, value) in enumerate(zip(TI_MS, shared_ijv_curve(image_np))):
        curve_rows.append({'arm': name, 'frame': frame, 'TI_ms': float(ti_ms),
                           'shared_IJV_magnitude': float(value)})
curve_table = pd.DataFrame(curve_rows)
curve_local = LOCAL / 'P2_shared_IJV_curves.csv'
curve_table.to_csv(curve_local, index=False)
copy_atomic(curve_local, DRIVE_RESULTS / curve_local.name)

if not oracle_row['common_image_gate_pass']:
    status = 'RANK5_MODEL_GAP_DETECTED'
elif t1_row['common_image_gate_pass']:
    status = 'RANK5_RETROSPECTIVE_GATE_PASS'
elif f0_row['common_image_gate_pass']:
    status = 'RANK5_RECON_OPTIMIZATION_OR_ENCODING_GAP'
else:
    status = 'RETROSPECTIVE_RECON_INCONCLUSIVE'

decision = {
    'experiment_id': EXPERIMENT_ID,
    'status': status,
    'rank5_oracle_gate_pass': bool(oracle_row['common_image_gate_pass']),
    'R5_T1_DCF_gate_pass': bool(t1_row['common_image_gate_pass']),
    'F0_noT1_DCF_gate_pass': bool(f0_row['common_image_gate_pass']),
    'scores': comparison.reset_index().to_dict(orient='records'),
    'decision_logic': {
        'oracle_fails': 'hard rank-5 model bias is already above the locked image gate',
        'oracle_passes_T1_fails_F0_passes': 'rank-5 model is adequate but its INR reconstruction/encoding is not',
        'T1_passes': 'advance to measured GASSP1 seed 0 and the real forward-model gate',
        'both_recons_fail_after_oracle_passes': 'do not identify the prior; diagnose optimization/capacity first',
    },
    'claim_boundary': 'real fully sampled image dynamics with synthetic GASSP1 k-space; no measured-k-space, noise, or B0-forward mismatch',
}
write_json_atomic(DRIVE_RESULTS / 'P2_retrospective_decision.json', decision)

target_energy = np.sum(np.abs(target_image_np) ** 2, axis=(1, 2))
null_frame = int(np.argmin(target_energy))
show_frames = list(dict.fromkeys([0, 12, null_frame, 25, 49]))
series = [('target', target_image_np), ('rank-5 oracle', oracle_image_np),
          ('R5 T1+DCF', t1_image_np), ('F0 noT1+DCF', f0_image_np)]
vmax = float(np.quantile(np.abs(target_image_np)[:, support], 0.995))
fig, axes = plt.subplots(len(series), len(show_frames),
                         figsize=(3 * len(show_frames), 2.8 * len(series)),
                         constrained_layout=True, squeeze=False)
for row, (label, image_value) in enumerate(series):
    for column, frame in enumerate(show_frames):
        axes[row, column].imshow(np.abs(image_value[frame]), cmap='gray', vmin=0, vmax=vmax)
        axes[row, column].set_title(f'{label}\nTI={TI_MS[frame]:.0f} ms')
        axes[row, column].axis('off')
figure_path = DRIVE_RESULTS / 'P2_selected_frames.png'
fig.savefig(figure_path, dpi=150, bbox_inches='tight')
plt.show()

fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
for name, rows in per_frame.groupby('arm'):
    axis.plot(rows['TI_ms'], rows['magnitude_NRMSE'], label=name)
axis.axhline(GATES['magnitude_NRMSE'], color='k', linestyle='--', linewidth=1, label='gate')
axis.set_xlabel('TI (ms)'); axis.set_ylabel('per-frame magnitude NRMSE')
axis.grid(True); axis.legend()
fig.savefig(DRIVE_RESULTS / 'P2_per_frame_magnitude_NRMSE.png', dpi=160, bbox_inches='tight')
plt.show()

fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
for name, rows in curve_table.groupby('arm'):
    values = rows['shared_IJV_magnitude'].to_numpy()
    axis.plot(rows['TI_ms'], values / max(np.linalg.norm(values), 1e-12), label=name)
axis.set_xlabel('TI (ms)'); axis.set_ylabel('L2-normalized shared-IJV magnitude')
axis.grid(True); axis.legend()
fig.savefig(DRIVE_RESULTS / 'P2_shared_IJV_curves.png', dpi=160, bbox_inches='tight')
plt.show()

try:
    font = ImageFont.truetype('DejaVuSans.ttf', 12)
except OSError:
    font = ImageFont.load_default()

def gray_panel(magnitude):
    values = np.uint8(np.rint(255 * np.clip(magnitude / vmax, 0, 1)))
    return Image.fromarray(values, mode='L').convert('RGB')

gif_frames = []
for frame in range(50):
    canvas = Image.new('RGB', (N * len(series), N + 42), 'black')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'TI={TI_MS[frame]:.0f} ms | fixed window', font=font, fill='white')
    for column, (label, image_value) in enumerate(series):
        left = N * column
        draw.text((left + 5, 23), label, font=font, fill='white')
        canvas.paste(gray_panel(np.abs(image_value[frame])), (left, 42))
    gif_frames.append(canvas)
gif_path = DRIVE_RESULTS / 'P2_comparison.gif'
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                   duration=200, loop=0, disposal=2, optimize=False)

combined_local = LOCAL / 'P2_retrospective_images.mat'
io.savemat(combined_local, {
    'target_image_normalized': target_image_np,
    'rank5_oracle_projection': oracle_image_np,
    'R5_T1_DCF': t1_image_np,
    'F0_noT1_DCF': f0_image_np,
    'basis': basis,
    'TI_ms': TI_MS,
    'target_scale': target_scale,
    'support': support,
})
copy_atomic(combined_local, DRIVE_RESULTS / combined_local.name)
print(decision)
"""
)

markdown("## 10. Final provenance")

code(
    r"""
manifest = {
    'experiment_id': EXPERIMENT_ID,
    'protocol_revision': PROTOCOL_REVISION,
    'runtime_signature': RUNTIME_SIGNATURE,
    'input_sha256': input_hashes,
    'basis_manifest': basis_manifest,
    'model_configs': MODEL_CONFIGS,
    'parameter_counts': parameter_counts,
    'training': {
        'uniform_parent_steps': UNIFORM_PARENT_STEPS,
        'dcf_continuation_steps': DCF_CONTINUATION_STEPS,
        'seed': SEED, 'lr': LR, 'weight_decay': 0.0,
        'checkpoint_selection': 'own k-space objective only',
    },
    'target_operator_audit': target_audit,
    'decision': decision,
    'outputs': sorted(path.name for path in DRIVE_RESULTS.iterdir()),
}
write_json_atomic(DRIVE_RESULTS / 'experiment_manifest.json', manifest)
print('Result folder:', DRIVE_RESULTS)
print('Summary:', DRIVE_RESULTS / 'P2_retrospective_summary.csv')
print('Decision:', DRIVE_RESULTS / 'P2_retrospective_decision.json')
print('Status:', decision['status'])
"""
)


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "name": ""},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n")

# Small builder self-check: syntax plus the scientific contract that is easy to regress.
for item in cells:
    if item["cell_type"] != "code":
        continue
    source = "".join(item["source"])
    python_source = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("!")
    )
    ast.parse(python_source)

all_source = "\n".join("".join(item["source"]) for item in cells)
training_source = next(
    "".join(item["source"]) for item in cells
    if item["cell_type"] == "code" and "def train_stage" in "".join(item["source"])
)
assert "del spiral['kdata']" in all_source
assert "full_spiral_preb0_target.mat" in all_source
assert "rank5_oracle_projection" in all_source
assert "R5_T1_DCF" in all_source and "F0_noT1_DCF" in all_source
assert "UNIFORM_PARENT_STEPS = 800" in all_source
assert "DCF_CONTINUATION_STEPS = 1600" in all_source
assert "weight_decay" not in training_source.split("# weight_decay=0")[0]
assert "own k-space objective only; target never used" in all_source
assert "measured GASSP1 k-space values" in all_source
assert "reference_TI_ms" in all_source and "50 + 200 * np.arange(50)" in all_source
assert len(cells) == 21, len(cells)

print(f"Wrote {OUTPUT} ({len(cells)} cells, {OUTPUT.stat().st_size} bytes)")
