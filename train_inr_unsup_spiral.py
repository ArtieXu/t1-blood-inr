"""
train_inr_unsup_spiral.py -- target-free (no CG) spiral INR.

Feng's model is kept exactly: 3D (t,y,x) HashGrid + MLP, base INR.train(), and
the released relative-L2 / temporal-TV / nuclear-norm loss family. There is no
latent modulation and no CG image anywhere in this file.

Only the in-vivo handling follows INMR:
  - the intensity scale comes from the measured-data adjoint, not from a target;
  - DCF can act as a gradient preconditioner instead of a term in the objective.
    INMR passes `density` to mri-nufft, which applies it inside adj_op; with
    autograd-through-op that means the density reaches the backward pass only.
    --dcf_backward reproduces that placement with torchkbnufft.

Coil combination is conj(S) (Feng and INMR both). Sinv is MATLAB-CG parity only
and is deliberately not wired in here.

Arms used for the DCF-placement question:
  U0  --dc_form feng_rel   --dc_weighting uniform                  (no DCF at all)
  U1  --dc_form feng_rel   --dc_weighting uniform  --dcf_backward  (DCF = preconditioner)
  U2  --dc_form global_rel --dc_weighting dcf                      (DCF in the objective)
"""
import os
import sys
import argparse
import datetime
import hashlib
import json
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1600)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-t', '--tv_weight', type=float, default=0.02)
parser.add_argument('-st', '--stv_weight', type=float, default=0.0)
parser.add_argument('-l', '--lr_weight', type=float, default=0.0002)
parser.add_argument('--dc_form', choices=['feng_rel', 'global_rel'], default='feng_rel',
                    help='feng_rel = released per-element predicted Re/Im normalization; '
                         'global_rel = weighted residual energy / weighted measured energy')
parser.add_argument('--dc_weighting', choices=['uniform', 'dcf'], default='uniform',
                    help='sample weighting inside the objective; only used by global_rel')
parser.add_argument('--dcf_backward', action='store_true',
                    help='INMR placement: leave the objective unweighted and apply the DCF in '
                         'the adjoint used by backward (preconditioner, not a new objective)')
parser.add_argument('--dcf_norm', choices=['none', 'mean'], default='mean',
                    help="mean keeps the preconditioned gradient on the unweighted scale; the "
                         "global DCF scale cancels in global_rel but not in --dcf_backward")
parser.add_argument('--eps', type=float, default=1e-4, help='relative-L2 denominator epsilon')
parser.add_argument('--n_levels', type=int, default=16)
parser.add_argument('--base_resolution', type=int, default=16)
parser.add_argument('-hs', '--log2_hashmap_size', type=int, default=24)
parser.add_argument('--hash_schedule', choices=['match_n', 'feng', 'inmr'], default='match_n',
                    help='finest hash resolution: N (all previous runs), Feng 2.0/level, or INMR 4N')
parser.add_argument('-ls', '--per_level_scale', type=float, default=None, help='overrides --hash_schedule')
parser.add_argument('-n', '--neuron', type=int, default=128)
parser.add_argument('-ly', '--layers', type=int, default=5)
parser.add_argument('-m', '--mask', action='store_true', help='coarse-to-fine hash levels (Feng -m)')
parser.add_argument('--warmup_levels', type=int, default=None,
                    help='number of hash levels active at epoch 1; requires --ctf_epochs')
parser.add_argument('--ctf_epochs', type=int, default=None,
                    help='epoch by which all hash levels are active; decouples coarse-to-fine from total epochs')
parser.add_argument('--checkpoint_active_levels', action='store_true',
                    help='render intermediate checkpoints with only the levels trained at that epoch')
parser.add_argument('--time_coords', choices=['ti_span', 'feng_literal'], default='ti_span',
                    help='ti_span spreads the 50 TI over [0,1]; feng_literal takes the first '
                         'frames points of an N-point lattice as the released code does')
parser.add_argument('--kb_grid_size', type=int, default=None, help='parity-validated value is 324')
parser.add_argument('--no_support_mask', action='store_true')
parser.add_argument('--scale_quantile', type=float, default=0.995)
parser.add_argument('--summary_epoch', type=int, default=100)
parser.add_argument('--data_path', type=str, default='gassp1_data.mat')
parser.add_argument('--temporal_basis_path', type=str, default=None,
                    help='optional orthonormal .npy basis [frames, rank]; projects every '
                         'reconstruction onto this temporal subspace before loss and saving')
parser.add_argument('--holdout_every', type=int, default=0,
                    help='withhold every Nth frame from DC loss; 0 uses every frame')
parser.add_argument('--holdout_offset', type=int, default=0,
                    help='zero-based remainder withheld when --holdout_every is enabled')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shared_init_path', type=str, default=None,
                    help='create or load one initial network state shared by controlled arms')
parser.add_argument('--ckpt_every', type=int, default=0,
                    help='periodically save a resumable checkpoint every N steps; '
                         '0 disables it and reproduces the pre-resume behaviour exactly')
parser.add_argument('--resume', type=str, default=None,
                    help='path to an existing ./log/<tag>_<ts> directory; continues that run '
                         'in place from its ckpt.pt instead of starting a new one')
parser.add_argument('--tag', type=str, default='inr_unsup')
parser.add_argument('--wandb', action='store_true', help='mirror loss.csv and run_info to Weights & Biases')
parser.add_argument('--wandb_project', type=str, default='t1-blood-inr')
parser.add_argument('--wandb_run', type=str, default=None, help='defaults to the log folder name')
args = parser.parse_args()
if min(args.tv_weight, args.stv_weight, args.lr_weight) < 0:
    parser.error('regularizer weights must be nonnegative')
if args.holdout_every == 1 or args.holdout_every < 0:
    parser.error('--holdout_every must be 0 or at least 2')
if args.holdout_every == 0 and args.holdout_offset != 0:
    parser.error('--holdout_offset requires --holdout_every')
if args.holdout_every and not 0 <= args.holdout_offset < args.holdout_every:
    parser.error('--holdout_offset must be in [0, holdout_every)')
if args.ctf_epochs is not None:
    if not args.mask:
        parser.error('--ctf_epochs requires -m/--mask')
    if not 0 < args.ctf_epochs < args.epochs:
        parser.error('--ctf_epochs must be between 1 and epochs - 1')
    if args.warmup_levels is None:
        parser.error('--ctf_epochs requires --warmup_levels')
if args.warmup_levels is not None:
    if args.ctf_epochs is None:
        parser.error('--warmup_levels requires --ctf_epochs')
    if not 1 <= args.warmup_levels <= args.n_levels:
        parser.error('--warmup_levels must be between 1 and --n_levels')

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# tiny-cuda-nn allocates through its own cuMemCreate arena, outside PyTorch's
# caching allocator. Expandable segments keep the torch side from reserving
# fragmented blocks that tcnn can then no longer obtain (CUDA_ERROR_OUT_OF_MEMORY
# from gpu_memory.h). Numerics are unaffected; an explicit env var still wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from tqdm import tqdm
from scipy import io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_spiral import load_spiral_data
from spiral_nufft import SpiralNUFFT
from model import INR
from utils import visual_mag, path_checker

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TI_START_MS = 63  # GASSP1 ExamCard/readout timing: 200 ms phase interval - 138 ms pulse timing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_level_scale_for(schedule, grid_size, base_resolution, n_levels):
    if n_levels <= 1:
        return 1.0
    if schedule == 'feng':
        return 2.0
    finest = grid_size if schedule == 'match_n' else 4 * grid_size
    return float((finest / base_resolution) ** (1 / (n_levels - 1)))


def time_coords(frame_num, mode, grid_size):
    if mode == 'feng_literal':
        t = torch.linspace(1 / (2 * grid_size), 1 - 1 / (2 * grid_size), grid_size)
        return t[:frame_num]
    ti_ms = TI_START_MS + torch.arange(frame_num, dtype=torch.float32) * 200
    t = (ti_ms - ti_ms.min()) / (ti_ms.max() - ti_ms.min())
    eps = 1 / (2 * frame_num)
    return eps + (1 - 2 * eps) * t


def robust_scale(x, q):
    return torch.quantile(x.abs().reshape(-1), min(max(float(q), 0.0), 1.0)).clamp_min(1e-8)


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def render(inr, pos, epoch, use_c2f_mask):
    img = torch.view_as_complex(
        inr.forward(pos, epoch, mask=(inr.mask and use_c2f_mask)).to(torch.float32).reshape(
            1, inr.nufft_op.grid_size, inr.nufft_op.grid_size, inr.nufft_op.frame_num, 2
        )
    ).squeeze(-1).permute(3, 0, 1, 2)
    if getattr(inr.nufft_op, 'support_mask', None) is not None:
        img = img * inr.nufft_op.support_mask
    img = project_temporal(img, getattr(inr, 'temporal_basis', None))
    return img


def project_temporal(img, basis):
    if basis is None:
        return img
    flat = img.reshape(img.shape[0], -1)
    basis = basis.to(dtype=flat.dtype)
    return (basis @ (basis.T @ flat)).reshape_as(img)


class _PrecondForward(torch.autograd.Function):
    """A in forward, A^H W in backward -- the INMR/mri-nufft density placement."""

    @staticmethod
    def forward(ctx, img, op):
        ctx.op = op
        with torch.no_grad():
            return op.forward(img)

    @staticmethod
    def backward(ctx, grad_k):
        return ctx.op.adjoint(grad_k, weighted=True), None


class PrecondNUFFT:

    def __init__(self, op):
        self._op = op

    def __getattr__(self, name):
        return getattr(self._op, name)

    def forward(self, img):
        return _PrecondForward.apply(img, self._op)


class UnsupINR(INR):
    """Feng's INR with a selectable DC term. Everything else is inherited."""

    def __init__(self, nufft_op, params, lr, eps, dc_form, loss_weight, kdata_power,
                 loss_frame_mask, temporal_basis=None):
        super(UnsupINR, self).__init__(nufft_op, params, lr, eps)
        self.dc_form = dc_form
        self.loss_weight = loss_weight
        self.kdata_power = kdata_power
        self.loss_frame_mask = loss_frame_mask
        self.temporal_basis = temporal_basis
        self.kpred = None

    def train(self, pos, kdata, e):
        timepoint = time.time()
        self.encoding.train()
        self.model.train()
        intensity = torch.view_as_complex(
            self.forward(pos, e, mask=self.mask).to(torch.float32).reshape(
                1, self.nufft_op.grid_size, self.nufft_op.grid_size,
                self.nufft_op.frame_num, 2
            )
        ).squeeze(-1).permute(3, 0, 1, 2)
        if getattr(self.nufft_op, 'support_mask', None) is not None:
            intensity = intensity * self.nufft_op.support_mask
        intensity = project_temporal(intensity, self.temporal_basis)
        kdata_sample = self.nufft_op.forward(intensity).reshape(
            self.nufft_op.frame_num, self.nufft_op.coil_num,
            self.nufft_op.spoke_num, self.nufft_op.spoke_length
        )
        self.loss_train = self.cal_loss(intensity, kdata_sample, kdata)
        self.optimizer.zero_grad()
        self.loss_train.backward()
        self.optimizer.step()
        self.scheduler.step()
        return intensity, time.time() - timepoint

    def cal_loss(self, intensity, kdata_sample, kdata):
        mx = torch.abs(intensity.detach()).max().clamp_min(1e-8)
        self.tv_loss = (self.TV_loss(intensity.real) + self.TV_loss(intensity.imag)) / mx
        self.stv_loss = (self.STV_loss(intensity.real) + self.STV_loss(intensity.imag)) / mx
        if self.lr_weight == 0:
            self.lowrank_loss = intensity.real.new_zeros(())
        else:
            self.lowrank_loss = self.LR_loss(intensity)
        predicted = kdata_sample[self.loss_frame_mask]
        measured = kdata[self.loss_frame_mask]
        if self.dc_form == 'feng_rel':
            self.dc_loss = self.DC_loss(predicted, measured).mean()
        else:
            err = predicted - measured
            weight = self.loss_weight[self.loss_frame_mask]
            self.dc_loss = (weight * (err.real ** 2 + err.imag ** 2)).sum() / self.kdata_power
        self.kpred = kdata_sample.detach()
        return (self.dc_loss + self.tv_weight * self.tv_loss
                + self.lr_weight * self.lowrank_loss + self.stv_weight * self.stv_loss)


set_seed(args.seed)

data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(HERE, args.data_path)
d = load_spiral_data(mat_path=data_path, apply_shift=False)
N = d['N']
ktraj = torch.as_tensor(d['ktraj'])
smap = torch.as_tensor(d['smap'])
wi = torch.as_tensor(d['wi'])
support_mask = None if args.no_support_mask else torch.as_tensor(d['mask']).to(torch.float32)
kdata_raw = torch.as_tensor(d['kdata']).to(device).to(torch.complex64)
frames = kdata_raw.shape[0]
frame_ids = torch.arange(frames, device=device)
if args.holdout_every:
    holdout_mask = frame_ids.remainder(args.holdout_every) == args.holdout_offset
else:
    holdout_mask = torch.zeros(frames, dtype=torch.bool, device=device)
train_mask = ~holdout_mask
if not torch.any(train_mask):
    raise ValueError('frame split leaves no training frames')
if args.holdout_every and not torch.any(holdout_mask):
    raise ValueError('frame split leaves no held-out frames')

temporal_basis = None
temporal_basis_path = None
temporal_basis_sha256 = None
if args.temporal_basis_path is not None:
    temporal_basis_path = (args.temporal_basis_path if os.path.isabs(args.temporal_basis_path)
                           else os.path.join(HERE, args.temporal_basis_path))
    basis_np = np.load(temporal_basis_path).astype(np.float32)
    if basis_np.ndim != 2 or basis_np.shape[0] != frames:
        raise ValueError('temporal basis must have shape [frames, rank]')
    if not np.isfinite(basis_np).all():
        raise ValueError('temporal basis contains non-finite values')
    gram = basis_np.T @ basis_np
    if not np.allclose(gram, np.eye(basis_np.shape[1]), atol=1e-4, rtol=1e-4):
        raise ValueError('temporal basis columns must be orthonormal')
    temporal_basis = torch.as_tensor(basis_np, device=device)
    temporal_basis_sha256 = sha256(temporal_basis_path)

base_op = SpiralNUFFT(
    ktraj, smap, wi, N, device,
    sinv=None,                       # conj(S) only; Sinv stays a MATLAB-parity control
    support_mask=support_mask,
    dcf_norm=args.dcf_norm,
    kb_grid_size=args.kb_grid_size,
    n_shift=(N / 2 + d['shift'][1], N / 2 + d['shift'][0]),
)

# Intensity scale from the measured data alone: the DCF-weighted Hermitian
# adjoint image (gridding recon). Uses the raw DCF so every arm shares one data
# normalization regardless of --dcf_norm; relative-L2 is not scale invariant.
with torch.no_grad():
    adj_raw = base_op.adj_op(
        kdata_raw * base_op.wi_raw.unsqueeze(1), base_op.ktraj, smaps=base_op.smap
    ) / base_op.grid_size
    scale = robust_scale(adj_raw[train_mask], args.scale_quantile)
kdata = (kdata_raw / scale).reshape(frames, base_op.coil_num, base_op.spoke_num, base_op.spoke_length)

nufft_op = PrecondNUFFT(base_op) if args.dcf_backward else base_op

use_dcf_in_loss = args.dc_weighting == 'dcf'
loss_weight = (base_op.wi if use_dcf_in_loss else torch.ones_like(base_op.wi)).reshape(frames, 1, 1, -1)
diag_weight = base_op.wi.reshape(frames, 1, 1, -1)
with torch.no_grad():
    if torch.any(base_op.wi < 0):
        raise ValueError('DCF weights must be nonnegative')
    kdata_power = (loss_weight[train_mask]
                   * (kdata[train_mask].real ** 2 + kdata[train_mask].imag ** 2)).sum()
    kdata_energy = (kdata.real ** 2 + kdata.imag ** 2)
    if kdata_power <= 0:
        raise ValueError('measured k-space must have positive energy')

per_level_scale = args.per_level_scale
if per_level_scale is None:
    per_level_scale = per_level_scale_for(args.hash_schedule, N, args.base_resolution, args.n_levels)

params = {
    'n_levels': args.n_levels,
    'n_features_per_level': 2,
    'log2_hashmap_size': args.log2_hashmap_size,
    'base_resolution': args.base_resolution,
    'per_level_scale': per_level_scale,
    'lr': args.lr,
    'n_neurons': args.neuron,
    'n_hidden_layers': args.layers,
    'tv_weight': args.tv_weight,
    'lr_weight': args.lr_weight,
    'stv_weight': args.stv_weight,
    'epochs': args.epochs,
    'mask': args.mask,
    'warmup_levels': args.warmup_levels,
    'ctf_epochs': args.ctf_epochs,
    'relL2': args.dc_form == 'feng_rel',
}
inr = UnsupINR(
    nufft_op, params, args.lr, args.eps, args.dc_form, loss_weight, kdata_power,
    train_mask, temporal_basis
)

architecture = {
    'n_levels': args.n_levels,
    'n_features_per_level': params['n_features_per_level'],
    'log2_hashmap_size': args.log2_hashmap_size,
    'base_resolution': args.base_resolution,
    'per_level_scale': per_level_scale,
    'n_neurons': args.neuron,
    'n_hidden_layers': args.layers,
}
shared_init_path = None
shared_init_action = 'seed_only'
shared_init_sha256 = None
if args.shared_init_path is not None:
    shared_init_path = os.path.abspath(os.path.expanduser(args.shared_init_path))
    if os.path.exists(shared_init_path):
        checkpoint = torch.load(shared_init_path, map_location=device)
        if checkpoint.get('architecture') != architecture:
            raise ValueError('shared initial state architecture does not match this run')
        inr.encoding.load_state_dict(checkpoint['encoding'])
        inr.model.load_state_dict(checkpoint['model'])
        shared_init_action = 'loaded'
    else:
        os.makedirs(os.path.dirname(shared_init_path), exist_ok=True)
        torch.save({
            'encoding': inr.encoding.state_dict(),
            'model': inr.model.state_dict(),
            'architecture': architecture,
            'seed': args.seed,
        }, shared_init_path)
        shared_init_action = 'created'
    shared_init_sha256 = sha256(shared_init_path)

ts = time_coords(frames, args.time_coords, N)
pos = inr.build_pos(N, frames, time_coords=ts)

start_epoch = 0
resume_time_usage = 0.0
if args.resume:
    log_path = args.resume if os.path.isabs(args.resume) else os.path.abspath(args.resume)
    ckpt_path = os.path.join(log_path, 'ckpt.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError('no ckpt.pt in {}'.format(log_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get('architecture') != architecture:
        raise ValueError('checkpoint architecture does not match this run')
    inr.encoding.load_state_dict(ckpt['encoding'])
    inr.model.load_state_dict(ckpt['model'])
    inr.optimizer.load_state_dict(ckpt['optimizer'])
    inr.scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = int(ckpt['epoch'])
    resume_time_usage = float(ckpt.get('time_usage', 0.0))
    # 恢复 RNG 状态，续跑才和不中断的一次跑等价
    random.setstate(ckpt['rng_python'])
    np.random.set_state(ckpt['rng_numpy'])
    torch.set_rng_state(ckpt['rng_torch'].cpu())
    if torch.cuda.is_available() and ckpt.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all([t.cpu() for t in ckpt['rng_cuda']])
    # 崩溃时 loss.csv 往往比 ckpt 更靠前（ckpt 是周期性写的），
    # 那些多出来的行对应的权重已经丢了。直接追加会产生重复且不连续的曲线，
    # 所以先把 loss.csv 截断到 checkpoint 的步数。
    _csv = os.path.join(log_path, 'loss.csv')
    if os.path.isfile(_csv):
        with open(_csv) as _fh:
            _lines = _fh.readlines()
        _head, _rows = _lines[:1], _lines[1:]
        _keep = [r for r in _rows if r.split(',')[0].isdigit() and int(r.split(',')[0]) <= start_epoch]
        if len(_keep) != len(_rows):
            print('truncating loss.csv: {} -> {} rows (checkpoint is at step {})'.format(
                len(_rows), len(_keep), start_epoch))
            with open(_csv, 'w') as _fh:
                _fh.writelines(_head + _keep)
    print('resumed {} at step {}'.format(log_path, start_epoch))
else:
    log_path = './log/{}_{}'.format(args.tag, datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    path_checker(log_path)


def save_ckpt(epoch, time_usage):
    """原子写：先写临时文件再 rename，中途断电不会留下半个损坏的 ckpt。"""
    tmp = os.path.join(log_path, 'ckpt.pt.tmp')
    torch.save({
        'encoding': inr.encoding.state_dict(),
        'model': inr.model.state_dict(),
        'optimizer': inr.optimizer.state_dict(),
        'scheduler': inr.scheduler.state_dict(),
        'epoch': epoch,
        'time_usage': time_usage,
        'architecture': architecture,
        'params': params,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp)
    os.replace(tmp, os.path.join(log_path, 'ckpt.pt'))

n_edge = max(1, base_op.spoke_length // 10)
run_info = {
    'script': 'train_inr_unsup_spiral.py',
    'cg_supervision': False,
    'target_free': True,
    'dc_form': args.dc_form,
    'dc_weighting_in_loss': args.dc_weighting,
    'dcf_in_backward': bool(args.dcf_backward),
    'dcf_norm': args.dcf_norm,
    'coil_backward': 'conj(S) autograd',
    'eps': args.eps,
    'learning_rate': args.lr,
    'tv_weight': args.tv_weight,
    'stv_weight': args.stv_weight,
    'lowrank_weight': args.lr_weight,
    'epochs': args.epochs,
    'frames': frames,
    'coils': base_op.coil_num,
    'samples': base_op.spoke_length,
    'grid_size': N,
    'kb_grid_size': args.kb_grid_size,
    'support_mask': support_mask is not None,
    'coarse_to_fine': bool(args.mask),
    'n_levels': args.n_levels,
    'base_resolution': args.base_resolution,
    'log2_hashmap_size': args.log2_hashmap_size,
    'n_neurons': args.neuron,
    'n_hidden_layers': args.layers,
    'epochs_per_level': inr.epochs_per_level,
    'warmup_levels': args.warmup_levels,
    'ctf_epochs': args.ctf_epochs,
    'checkpoint_active_levels': bool(args.checkpoint_active_levels),
    'hash_schedule': args.hash_schedule,
    'per_level_scale': per_level_scale,
    'finest_resolution': float(args.base_resolution * per_level_scale ** (args.n_levels - 1)),
    'time_coords': args.time_coords,
    'time_coords_range': [float(ts.min()), float(ts.max())],
    'ti_ms': [TI_START_MS + 200 * i for i in range(frames)],
    'scale_source': 'quantile{} of |A^H W y| over training frames'.format(args.scale_quantile),
    'scale': float(scale),
    'wi_raw_mean': float(base_op.wi_raw.mean()),
    'wi_used_mean': float(base_op.wi.mean()),
    'seed': args.seed,
    'temporal_model': ('flexible_3d_hash' if temporal_basis is None
                       else 'hard_linear_subspace'),
    'temporal_basis_path': temporal_basis_path,
    'temporal_basis_sha256': temporal_basis_sha256,
    'temporal_basis_rank': None if temporal_basis is None else int(temporal_basis.shape[1]),
    'holdout_every': args.holdout_every,
    'holdout_offset': args.holdout_offset,
    'train_frames_zero_based': frame_ids[train_mask].cpu().tolist(),
    'holdout_frames_zero_based': frame_ids[holdout_mask].cpu().tolist(),
    'shared_init_path': shared_init_path,
    'shared_init_action': shared_init_action,
    'shared_init_sha256': shared_init_sha256,
    'data_path': os.path.abspath(data_path),
    'ckpt_every': args.ckpt_every,
    'resumed_from_step': start_epoch if args.resume else None,
}
with open(os.path.join(log_path,
          'run_info_resume_{}.json'.format(start_epoch) if args.resume else 'run_info.json'), 'w') as f:
    json.dump(run_info, f, indent=2)
print(json.dumps(run_info, indent=2))

wandb_run = None
if args.wandb:
    import wandb
    wandb_run = wandb.init(project=args.wandb_project,
                           name=args.wandb_run or os.path.basename(log_path),
                           config=run_info)


def gpu_peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


loss_csv = os.path.join(log_path, 'loss.csv')
# 续跑时以追加方式打开，且不重写表头 —— 一次实验只有一份连续的 loss.csv
_append = bool(args.resume) and os.path.isfile(loss_csv)
with open(loss_csv, 'a' if _append else 'w') as f:
    if not _append:
        f.write('epoch,total,dc,tv,stv,lowrank,lr,time_s,'
                'dc_uniform_rel,dc_dcf_rel,dc_center_rel,dc_outer_rel,eps_frac,gpu_peak_gb,'
                'dc_train_uniform_rel,dc_holdout_uniform_rel,dc_train_dcf_rel,dc_holdout_dcf_rel\n')
    time_usage = resume_time_usage
    loop = tqdm(range(start_epoch, args.epochs), total=args.epochs - start_epoch,
                initial=0, leave=True)
    for e in loop:
        intensity, dt = inr.train(pos, kdata, e)
        time_usage += dt

        with torch.no_grad():
            err = inr.kpred - kdata
            err_energy = err.real ** 2 + err.imag ** 2
            dc_uniform_rel = err_energy.sum() / kdata_energy.sum()
            dc_dcf_rel = (diag_weight * err_energy).sum() / (diag_weight * kdata_energy).sum()
            dc_center_rel = err_energy[..., :n_edge].sum() / kdata_energy[..., :n_edge].sum()
            dc_outer_rel = err_energy[..., -n_edge:].sum() / kdata_energy[..., -n_edge:].sum()
            dc_train_uniform_rel = err_energy[train_mask].sum() / kdata_energy[train_mask].sum()
            dc_train_dcf_rel = ((diag_weight[train_mask] * err_energy[train_mask]).sum()
                                / (diag_weight[train_mask] * kdata_energy[train_mask]).sum())
            if torch.any(holdout_mask):
                dc_holdout_uniform_rel = (err_energy[holdout_mask].sum()
                                          / kdata_energy[holdout_mask].sum())
                dc_holdout_dcf_rel = ((diag_weight[holdout_mask] * err_energy[holdout_mask]).sum()
                                      / (diag_weight[holdout_mask]
                                         * kdata_energy[holdout_mask]).sum())
            else:
                dc_holdout_uniform_rel = err_energy.new_tensor(float('nan'))
                dc_holdout_dcf_rel = err_energy.new_tensor(float('nan'))
            eps_frac = 0.5 * ((inr.kpred.real ** 2 < args.eps).float().mean()
                              + (inr.kpred.imag ** 2 < args.eps).float().mean())

        cur_lr = inr.scheduler.get_last_lr()[0]
        peak_gb = gpu_peak_gb()
        f.write('{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.4f},'
                '{:.8e},{:.8e},{:.8e},{:.8e},{:.6f},{:.3f},'
                '{:.8e},{:.8e},{:.8e},{:.8e}\n'.format(
                    e + 1, inr.loss_train.item(), inr.dc_loss.item(), inr.tv_loss.item(),
                    inr.stv_loss.item(), inr.lowrank_loss.item(), cur_lr, time_usage,
                    dc_uniform_rel.item(), dc_dcf_rel.item(), dc_center_rel.item(),
                    dc_outer_rel.item(), eps_frac.item(), peak_gb,
                    dc_train_uniform_rel.item(), dc_holdout_uniform_rel.item(),
                    dc_train_dcf_rel.item(), dc_holdout_dcf_rel.item()))
        if (e + 1) % 10 == 0:
            f.flush()
        loop.set_description('[INR-unsup][Lr:{:.2e}]'.format(cur_lr))
        loop.set_postfix(dc=inr.dc_loss.item(), dc_rel=dc_uniform_rel.item(),
                         tv=inr.tv_loss.item(), eps=eps_frac.item(), peak=peak_gb)
        if wandb_run is not None:
            wandb_run.log({
                'total': inr.loss_train.item(), 'dc': inr.dc_loss.item(),
                'tv': inr.tv_loss.item(), 'stv': inr.stv_loss.item(),
                'lowrank': inr.lowrank_loss.item(), 'lr': cur_lr,
                'dc_uniform_rel': dc_uniform_rel.item(), 'dc_dcf_rel': dc_dcf_rel.item(),
                'dc_center_rel': dc_center_rel.item(), 'dc_outer_rel': dc_outer_rel.item(),
                'dc_train_uniform_rel': dc_train_uniform_rel.item(),
                'dc_holdout_uniform_rel': dc_holdout_uniform_rel.item(),
                'dc_train_dcf_rel': dc_train_dcf_rel.item(),
                'dc_holdout_dcf_rel': dc_holdout_dcf_rel.item(),
                'outer_over_center': (dc_outer_rel / dc_center_rel.clamp_min(1e-30)).item(),
                'eps_frac': eps_frac.item(), 'gpu_peak_gb': peak_gb,
                'time_s': time_usage,
            }, step=e + 1)

        if args.ckpt_every and (e + 1) % args.ckpt_every == 0 and (e + 1) < args.epochs:
            f.flush()
            save_ckpt(e + 1, time_usage)

        if (e + 1) % args.summary_epoch == 0:
            with torch.no_grad():
                save_img = render(inr, pos, e, use_c2f_mask=args.checkpoint_active_levels)
                io.savemat(os.path.join(log_path, 'recon_{}.mat'.format(e + 1)), {
                    'img_inr': save_img.cpu().numpy(),
                    'img_inr_physical': (save_img * scale).cpu().numpy(),
                    'scale': float(scale),
                })
                montage = os.path.join(log_path, 'recon_{}_abs.png'.format(e + 1))
                visual_mag(save_img, montage, nrow_num=10)
                if wandb_run is not None:
                    import wandb
                    wandb_run.log({'recon': wandb.Image(montage)}, step=e + 1)

with torch.no_grad():
    final_img = render(inr, pos, args.epochs - 1, use_c2f_mask=False)
    io.savemat(os.path.join(log_path, 'recon_final.mat'), {
        'img_inr': final_img.cpu().numpy(),
        'img_inr_physical': (final_img * scale).cpu().numpy(),
        'scale': float(scale),
    })
    visual_mag(final_img, os.path.join(log_path, 'recon_final_abs.png'), nrow_num=10)
    torch.save({
        'encoding': inr.encoding.state_dict(),
        'model': inr.model.state_dict(),
        'params': params,
        'run_info': run_info,
    }, os.path.join(log_path, 'final_state.pt'))

    final_kpred = base_op.forward(final_img).reshape(
        frames, base_op.coil_num, base_op.spoke_num, base_op.spoke_length
    )
    final_err_energy = ((final_kpred - kdata).real ** 2 + (final_kpred - kdata).imag ** 2)
    with open(os.path.join(log_path, 'final_residual_by_frame.csv'), 'w') as f:
        f.write('frame_zero_based,ti_ms,split,uniform_rel,dcf_rel,center_rel,outer_rel\n')
        for i in range(frames):
            data_i = kdata_energy[i]
            err_i = final_err_energy[i]
            dcf_i = diag_weight[i]
            uniform_rel = err_i.sum() / data_i.sum()
            dcf_rel = (dcf_i * err_i).sum() / (dcf_i * data_i).sum()
            center_rel = err_i[..., :n_edge].sum() / data_i[..., :n_edge].sum()
            outer_rel = err_i[..., -n_edge:].sum() / data_i[..., -n_edge:].sum()
            split = 'holdout' if bool(holdout_mask[i]) else 'train'
            f.write('{},{},{},{:.8e},{:.8e},{:.8e},{:.8e}\n'.format(
                i, TI_START_MS + 200 * i, split, uniform_rel.item(), dcf_rel.item(),
                center_rel.item(), outer_rel.item()))

# 正常跑完就把续跑点删掉，免得下次 --resume 误接一个已经完成的实验
if args.ckpt_every:
    for _p in (os.path.join(log_path, 'ckpt.pt'), os.path.join(log_path, 'ckpt.pt.tmp')):
        if os.path.isfile(_p):
            os.remove(_p)

print('log:', log_path)
print('peak GPU memory (GB): {:.3f}'.format(gpu_peak_gb()))
if wandb_run is not None:
    import wandb
    wandb_run.summary['gpu_peak_gb'] = gpu_peak_gb()
    wandb_run.summary['log_path'] = log_path
    wandb_run.log({'recon_final': wandb.Image(os.path.join(log_path, 'recon_final_abs.png'))})
    wandb_run.finish()
