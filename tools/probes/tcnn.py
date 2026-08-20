#!/usr/bin/env python3
"""
probes/tcnn.py -- actually exercise tiny-cuda-nn on this GPU.

WHY THIS EXISTS
verify_migration.py --env does `__import__('tinycudann')` and nothing more.
Importing tinycudann only proves the Python bindings loaded. It does NOT prove
that CUDA kernels were compiled for this GPU's architecture. RTX 5070 is
Blackwell sm_120; a tiny-cuda-nn built without sm_120 in its architecture list
imports cleanly and then fails (or silently degrades) the first time a kernel
actually runs. That first time would otherwise be inside the smoke test.

This builds the EXACT network train_inr_unsup_spiral.py builds -- same
HashGrid config, same FullyFusedMLP, same point count -- and runs a real
forward + backward + optimizer step.

    python3 probes/tcnn.py
    python3 probes/tcnn.py --grid_size 216 --frames 50

Exit 0 = tiny-cuda-nn genuinely runs on this device at full problem size.
"""
from __future__ import annotations

import argparse, sys, time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid_size", type=int, default=216, help="N, matches gassp1 image size")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--n_levels", type=int, default=16)
    ap.add_argument("--base_resolution", type=int, default=16)
    ap.add_argument("--log2_hashmap_size", type=int, default=24)
    ap.add_argument("--neuron", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available to torch")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__} cuda {torch.version.cuda}")

    try:
        import tinycudann as tcnn
    except Exception as exc:                                        # noqa: BLE001
        print(f"FAIL: cannot import tinycudann: {type(exc).__name__}: {exc}")
        return 1
    print(f"tinycudann imported: {getattr(tcnn, '__version__', 'version unknown')}")

    N, F = args.grid_size, args.frames
    per_level_scale = float((N / args.base_resolution) ** (1 / (args.n_levels - 1)))
    enc_cfg = {
        "otype": "HashGrid",
        "n_levels": args.n_levels,
        "n_features_per_level": 2,
        "log2_hashmap_size": args.log2_hashmap_size,
        "base_resolution": args.base_resolution,
        "per_level_scale": per_level_scale,
    }
    net_cfg = {
        "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
        "n_neurons": args.neuron, "n_hidden_layers": args.layers,
    }
    print(f"config: {enc_cfg}\n        {net_cfg}")

    torch.cuda.reset_peak_memory_stats()
    try:
        encoding = tcnn.Encoding(n_input_dims=3, encoding_config=enc_cfg)
        network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=2,
                               network_config=net_cfg)
    except Exception as exc:                                        # noqa: BLE001
        print(f"FAIL: constructing the tcnn modules raised "
              f"{type(exc).__name__}: {exc}")
        return 1
    n_enc = sum(p.numel() for p in encoding.parameters())
    n_net = sum(p.numel() for p in network.parameters())
    print(f"parameters: encoding {n_enc:,}  network {n_net:,}  total {n_enc + n_net:,}")
    if n_enc == 0:
        print("FAIL: the encoding reports zero parameters")
        return 1

    # exactly INR.build_pos: the full (t,y,x) lattice, all points every step
    xs = torch.linspace(1 / (2 * N), 1 - 1 / (2 * N), N, device="cuda")
    ts = torch.linspace(1 / (2 * F), 1 - 1 / (2 * F), F, device="cuda")
    xv, yv, tv = torch.meshgrid([xs, xs, ts], indexing="ij")
    pos = torch.stack((tv.flatten(), yv.flatten(), xv.flatten())).t()
    print(f"pos: {tuple(pos.shape)}  ({pos.shape[0]:,} points per step)")

    optimizer = torch.optim.Adam([
        {"params": network.parameters(), "lr": 1e-3, "weight_decay": 1e-6},
        {"params": encoding.parameters(), "lr": 1e-3, "weight_decay": 0},
    ])

    try:
        times = []
        for step in range(args.steps):
            torch.cuda.synchronize(); t0 = time.time()
            out = network(encoding(pos.reshape(-1, 3))).to(torch.float32)
            img = torch.view_as_complex(out.reshape(1, N, N, F, 2)).squeeze(-1).permute(3, 0, 1, 2)
            loss = (img.real ** 2 + img.imag ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            torch.cuda.synchronize(); times.append(time.time() - t0)
            finite = bool(torch.isfinite(out).all())
            print(f"  step {step + 1}: loss={loss.item():.6e}  finite={finite}  "
                  f"{times[-1] * 1000:.1f} ms")
            if not finite:
                print("FAIL: tcnn produced non-finite output")
                return 1
    except Exception as exc:                                        # noqa: BLE001
        print(f"FAIL: running the tcnn kernels raised {type(exc).__name__}: {exc}")
        print("      This is the failure `import tinycudann` cannot detect: the")
        print("      bindings load but no kernel exists for this architecture.")
        return 1

    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in list(encoding.parameters()) + list(network.parameters()))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\ngradients finite on every parameter: {grad_ok}")
    print(f"peak torch-allocated VRAM (INR half only): {peak:.3f} GiB")
    print(f"steady-state step time: {sum(times[1:]) / max(1, len(times) - 1) * 1000:.1f} ms")
    if not grad_ok:
        print("FAIL: some parameter has a missing or non-finite gradient")
        return 1
    print("\nPROBE PASSED: tiny-cuda-nn compiles, runs and back-propagates on this GPU")
    print("at the full problem size. NOTE this covers the INR half only; the NUFFT")
    print("half is exercised by the Gate B smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
