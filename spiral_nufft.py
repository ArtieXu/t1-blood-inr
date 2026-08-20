"""
spiral_nufft.py -- NUFFT operator for single-shot spiral, drop-in for model.INR.

Replaces the repo's utils.NUFFT (which hardcodes grid_size = spoke_length // 2,
a radial assumption). Here grid_size is explicit (=216), one arm per frame
(spoke_num=1), spoke_length = samples per arm (=3315). torchkbnufft handles the
arbitrary spiral trajectory; the forward model is identical to the authors'
A = NUFFT(I . S).
"""
import torch
import torchkbnufft as tkbn


class SpiralNUFFT:
    def __init__(
        self,
        ktraj,
        smap,
        wi,
        grid_size,
        device,
        sinv=None,
        support_mask=None,
        dcf_norm='none',
        kb_grid_size=None,
        n_shift=None,
        numpoints=6,
    ):
        # ktraj [frames,2,samples] float (radians), smap [coils,N,N] complex,
        # sinv [coils,N,N] complex pseudo-inverse maps for MATLAB-style adjoint,
        # wi [frames,samples] float density compensation.
        self.ktraj = ktraj.to(device).to(torch.float32)
        self.smap = smap.to(device).to(torch.complex64)
        self.sinv = None if sinv is None else sinv.to(device).to(torch.complex64)
        wi = wi.to(device).to(torch.float32)
        self.wi_raw = wi
        if dcf_norm == 'none':
            self.wi = wi
        elif dcf_norm == 'mean':
            self.wi = wi / wi.mean().clamp_min(1e-12)
        else:
            raise ValueError("dcf_norm must be 'none' or 'mean'")
        self.grid_size = int(grid_size)
        self.frame_num = ktraj.shape[0]
        self.spoke_num = 1
        self.spoke_length = ktraj.shape[-1]
        self.coil_num = smap.shape[0]
        self.support_mask = None
        if support_mask is not None:
            mask = support_mask.to(device).to(torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            self.support_mask = mask
        im_size = (self.grid_size, self.grid_size)
        nufft_kwargs = {'im_size': im_size}
        if kb_grid_size is not None:
            if isinstance(kb_grid_size, int):
                kb_grid_size = (kb_grid_size, kb_grid_size)
            nufft_kwargs['grid_size'] = tuple(int(x) for x in kb_grid_size)
        if n_shift is not None:
            nufft_kwargs['n_shift'] = tuple(float(x) for x in n_shift)
        if numpoints is not None:
            if isinstance(numpoints, int):
                numpoints = (numpoints, numpoints)
            nufft_kwargs['numpoints'] = tuple(int(x) for x in numpoints)
        self.nufft_op = tkbn.KbNufft(**nufft_kwargs).to(torch.complex64).to(device)
        self.adj_op = tkbn.KbNufftAdjoint(**nufft_kwargs).to(torch.complex64).to(device)

    def forward(self, img):
        # img [frames,1,N,N] complex -> [frames,coils,samples]
        k = self.nufft_op(img, self.ktraj, smaps=self.smap)
        return k / self.grid_size

    def adjoint(self, kdata, weighted=True):
        # kdata [frames,coils,samples] -> Hermitian S-map-combined [frames,1,N,N].
        # This is useful for diagnostics, but not for GASSP1 scaling because the
        # MATLAB recon uses Sinv pseudo-inverse maps in the adjoint.
        wk = kdata * self.wi.unsqueeze(1) if weighted else kdata
        img = self.adj_op(wk, self.ktraj, smaps=self.smap)
        return img / self.grid_size

    def coil_adjoint(self, kdata, weighted=True):
        # kdata [frames,coils,samples] -> per-coil adjoint images [frames,coils,N,N].
        if kdata.ndim == 4:
            kdata = kdata.reshape(self.frame_num, self.coil_num, -1)
        wk = kdata * self.wi.unsqueeze(1) if weighted else kdata
        img = self.adj_op(wk, self.ktraj)
        return img / self.grid_size

    def pseudo_adjoint(self, kdata, weighted=True):
        # MATLAB-style coil combination: sum_c NUFFT'(k_c) * Sinv_c.
        if self.sinv is None:
            raise ValueError('pseudo_adjoint requires sinv maps')
        coil_img = self.coil_adjoint(kdata, weighted=weighted)
        return torch.sum(coil_img * self.sinv.unsqueeze(0), dim=1, keepdim=True)
