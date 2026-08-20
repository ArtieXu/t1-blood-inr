"""
CPU stand-in for tiny-cuda-nn, for pipeline dry-runs only.

It is SHAPE-FAITHFUL, not numerically equivalent: same constructor signatures,
same n_output_dims, same autograd behaviour, so every line of
train_inr_unsup_spiral.py / model.py downstream of the network sees exactly what
it would see on a GPU. It does NOT reproduce tiny-cuda-nn's hash encoding or
FullyFusedMLP kernels. Never use it for a scientific run.
"""
import torch

__version__ = "STUB-cpu (not tiny-cuda-nn)"


class Encoding(torch.nn.Module):
    def __init__(self, n_input_dims, encoding_config, **_):
        super().__init__()
        cfg = encoding_config
        self.n_levels = cfg["n_levels"]
        self.n_features_per_level = cfg["n_features_per_level"]
        self.base_resolution = cfg["base_resolution"]
        self.per_level_scale = cfg["per_level_scale"]
        self.n_output_dims = self.n_levels * self.n_features_per_level
        # one tiny linear map per level, standing in for a hash-grid lookup
        self.levels = torch.nn.ModuleList([
            torch.nn.Linear(n_input_dims, self.n_features_per_level)
            for _ in range(self.n_levels)])

    def forward(self, x):
        freqs = [self.base_resolution * self.per_level_scale ** i for i in range(self.n_levels)]
        return torch.cat([lin(torch.sin(f * x)) for lin, f in zip(self.levels, freqs)], dim=-1)


class Network(torch.nn.Module):
    def __init__(self, n_input_dims, n_output_dims, network_config, **_):
        super().__init__()
        w, d = network_config["n_neurons"], network_config["n_hidden_layers"]
        layers, prev = [], n_input_dims
        for _ in range(d):
            layers += [torch.nn.Linear(prev, w), torch.nn.ReLU()]
            prev = w
        layers += [torch.nn.Linear(prev, n_output_dims)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NetworkWithInputEncoding(torch.nn.Module):
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, **_):
        super().__init__()
        self.enc = Encoding(n_input_dims, encoding_config)
        self.net = Network(self.enc.n_output_dims, n_output_dims, network_config)

    def forward(self, x):
        return self.net(self.enc(x))
