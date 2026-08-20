# reports/

每个实验的**文本**结果。这是让协作者（包括 AI）看到本机跑了什么的唯一通道。

由 `tools/publish.sh <实验名>` 生成，内容：

| 文件 | 是什么 |
|---|---|
| `SUMMARY.md` | 契约 + 训练曲线抽样 + 验收结论，先看这个 |
| `run_info.json` | 训练脚本自己记的完整参数 |
| `loss.csv` | 逐步指标 |
| `final_residual_by_frame.csv` | 逐帧残差 |
| `acceptance.txt` | `tools/check_run.py` 的判定 |
| `sysinfo.json` | GPU / 驱动 / torch / CUDA / tcnn 版本 |
| `train_tail.log` | stdout 尾部，崩溃时看这个 |

**不会出现在这里**：`.mat`、`.pt`、`.png`。重建影像属于受试者数据，留在本地
`~/runs/<实验名>/`，由 `.gitignore` 挡住。

provenance 靠 git commit：某个结果是哪份代码跑的，看它所在的 commit 即可。
