# smoke

## 实验契约

| 项 | 值 |
|---|---|
| `epochs` | 32 |
| `seed` | 0 |
| `frames` | 50 |
| `coils` | 28 |
| `grid_size` | 216 |
| `kb_grid_size` | 324 |
| `dc_form` | feng_rel |
| `dc_weighting_in_loss` | uniform |
| `dcf_in_backward` | False |
| `dcf_norm` | mean |
| `time_coords` | ti_span |
| `temporal_model` | flexible_3d_hash |
| `temporal_basis_path` | None |
| `holdout_every` | 0 |
| `ckpt_every` | 0 |
| `resumed_from_step` | None |
| `scale` | 329.8946533203125 |

## 训练 32 步

| step | dc | dc_uniform_rel | eps_frac | peak GiB |
|---:|---:|---:|---:|---:|
| 1 | 6.8425e+04 | 1.0000 | 1.000 | 3.63 |
| 9 | 6.6454e+02 | 0.9051 | 0.808 | 4.31 |
| 17 | 9.3969e+01 | 0.6625 | 0.592 | 4.33 |
| 25 | 5.0255e+01 | 0.5480 | 0.513 | 4.33 |
| 32 | 3.2301e+01 | 0.4487 | 0.457 | 4.33 |

稳态 3.59 s/步，总计 2.0 分钟，显存峰值 4.33 GiB

## 验收

```
GATE B PASSED -- the 5070 stack runs, outputs are finite and correctly shaped.
```

> 大产物（.mat/.pt/.png）留在本地未入库；影像属于受试者数据。
