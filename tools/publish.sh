#!/usr/bin/env bash
# publish.sh -- 把一次实验的【文本】结果提交进仓库。
#
#   bash tools/publish.sh u0_1600
#   bash tools/publish.sh u0_1600 --push
#
# 这是唯一让协作者（包括 AI）看到你机器上跑了什么的通道：
# 结果进 git -> 对方 clone -> 直接读。
#
# 只带 loss.csv / run_info.json / acceptance.txt / final_residual_by_frame.csv / train.log 尾部。
# .mat / .pt / .png 一律留在本地 —— 重建影像属于受试者数据，绝不入库。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/.." && pwd)"
NAME="${1:?用法: bash tools/publish.sh <实验名> [--push]}"; shift || true
PUSH=0; for a in "$@"; do [ "$a" = "--push" ] && PUSH=1; done
SRC="${T1_RUNS:-${HOME}/runs}/${NAME}"
[ -d "${SRC}" ] || { echo "找不到 ${SRC}" >&2; exit 1; }
DST="${ROOT}/reports/${NAME}"; mkdir -p "${DST}"

LOGDIR="$(find "${SRC}/log" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
for f in loss.csv run_info.json final_residual_by_frame.csv; do
  [ -f "${LOGDIR}/${f}" ] && cp "${LOGDIR}/${f}" "${DST}/"
done
for f in "${LOGDIR}"/run_info_resume_*.json; do [ -f "$f" ] && cp "$f" "${DST}/"; done
[ -f "${SRC}/acceptance.txt" ] && cp "${SRC}/acceptance.txt" "${DST}/"
[ -f "${SRC}/train.log" ] && tail -c 200000 "${SRC}/train.log" > "${DST}/train_tail.log"
python3 "${HERE}/sysinfo.py" -o "${DST}/sysinfo.json" >/dev/null 2>&1

# 一眼能看懂的摘要，放最前面
python3 - "${DST}" "${NAME}" <<'PY' > "${DST}/SUMMARY.md"
import csv, json, os, sys
d, name = sys.argv[1], sys.argv[2]
print(f"# {name}\n")
ri = os.path.join(d, 'run_info.json')
if os.path.isfile(ri):
    r = json.load(open(ri))
    print("## 实验契约\n")
    print("| 项 | 值 |\n|---|---|")
    for k in ('epochs','seed','frames','coils','grid_size','kb_grid_size','dc_form',
              'dc_weighting_in_loss','dcf_in_backward','dcf_norm','time_coords',
              'temporal_model','temporal_basis_path','holdout_every','ckpt_every',
              'resumed_from_step','scale'):
        if k in r: print(f"| `{k}` | {r[k]} |")
lc = os.path.join(d, 'loss.csv')
if os.path.isfile(lc):
    rows = list(csv.DictReader(open(lc)))
    if rows:
        print(f"\n## 训练 {len(rows)} 步\n")
        print("| step | dc | dc_uniform_rel | eps_frac | peak GiB |\n|---:|---:|---:|---:|---:|")
        idx = sorted({0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1})
        for i in idx:
            x = rows[i]
            print(f"| {x['epoch']} | {float(x['dc']):.4e} | {float(x['dc_uniform_rel']):.4f} "
                  f"| {float(x['eps_frac']):.3f} | {float(x['gpu_peak_gb']):.2f} |")
        t = [float(x['time_s']) for x in rows]
        ps = [b-a for a, b in zip(t, t[1:])] or [0]
        print(f"\n稳态 {sum(ps[1:])/max(1,len(ps[1:])):.2f} s/步，总计 {t[-1]/60:.1f} 分钟，"
              f"显存峰值 {max(float(x['gpu_peak_gb']) for x in rows):.2f} GiB")
ac = os.path.join(d, 'acceptance.txt')
if os.path.isfile(ac):
    txt = open(ac).read()
    v = [l for l in txt.splitlines() if 'GATE B' in l or l.startswith('WARNING') or l.startswith('- ')]
    print("\n## 验收\n\n```\n" + "\n".join(v[:12]) + "\n```")
print("\n> 大产物（.mat/.pt/.png）留在本地未入库；影像属于受试者数据。")
PY

echo "已写入 reports/${NAME}/:"; ls -la "${DST}" | tail -n +4 | awk '{printf "  %8s  %s\n",$5,$9}'
BIG=$(find "${DST}" -type f -size +1M); [ -n "${BIG}" ] && { echo "警告：有超过 1 MB 的文件"; echo "${BIG}"; }
cd "${ROOT}"
git add "reports/${NAME}" 2>/dev/null
if [ "${PUSH}" -eq 1 ]; then
  git commit -q -m "results: ${NAME}" && git push -q && echo "已推送。协作者现在能看到了。"
else
  echo; echo "已暂存。提交并推送："
  echo "  git commit -m 'results: ${NAME}' && git push"
fi
