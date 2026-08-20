#!/usr/bin/env bash
# install_deps.sh -- 安全补上 Gate A 查出的 torchvision / scikit-image。
#
#   source tools/env.sh
#   bash   tools/install_deps.sh
#
# 为什么需要一个专门的脚本而不是直接 pip install：
# tinycudann 是编译型 C++ 扩展，链接的是 torch 2.7.1 的 ABI。裸跑
# `pip install torchvision` 很可能拉一个新版 torchvision，它会把 torch 一起升级
# —— 那样 tinycudann 立刻失效，你得重新编译一遍。
# 本脚本：先快照 torch 版本 -> dry-run 确认 pip 不会动 torch -> 安装 -> 复查
# torch 未变且 tcnn 内核仍能跑。任何一步不对就停下并告诉你怎么恢复。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TV_PIN="${TV_PIN:-0.22.1}"          # torch 2.7.1 <-> torchvision 0.22.1
IDX="https://download.pytorch.org/whl/cu128"

[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source env.sh" >&2; exit 2; }
echo "venv: ${VIRTUAL_ENV}"

snap() { python3 -c "import torch;print(torch.__version__, torch.version.cuda)" 2>/dev/null; }
BEFORE="$(snap)"; echo "安装前 torch: ${BEFORE}"
[ -n "${BEFORE}" ] || { echo "torch 都导不进来，先别装东西" >&2; exit 1; }

echo -e "\n[1/4] dry-run：确认 pip 不会碰 torch"
DRY="$(python3 -m pip install "torchvision==${TV_PIN}" --index-url "${IDX}" --dry-run 2>&1)"
echo "${DRY}" | tail -6
if echo "${DRY}" | grep -qiE "Would install .*\btorch-[0-9]"; then
  echo -e "\n中止：pip 打算连 torch 一起装/升级。那会打断 tinycudann 的 ABI。"
  echo "改用: python3 -m pip install --no-deps \"torchvision==${TV_PIN}\" --index-url ${IDX}"
  echo "（前提是 pillow 已装：python3 -c 'import PIL'）"
  exit 1
fi
echo "ok：torch 不在安装列表里"

echo -e "\n[2/4] 安装 torchvision==${TV_PIN}"
python3 -m pip install "torchvision==${TV_PIN}" --index-url "${IDX}" || exit 1

echo -e "\n[3/4] 安装 scikit-image（纯 Python 依赖，不涉及 torch）"
python3 -m pip install scikit-image || exit 1

echo -e "\n[4/4] 复查"
AFTER="$(snap)"; echo "安装后 torch: ${AFTER}"
if [ "${BEFORE}" != "${AFTER}" ]; then
  echo -e "\n!! torch 被改动了: ${BEFORE} -> ${AFTER}"
  echo "   tinycudann 很可能已经失效。恢复:"
  echo "   python3 -m pip install torch==${BEFORE%% *} --index-url ${IDX} --force-reinstall"
  exit 1
fi
python3 "${HERE}/deps.py" || exit 1
echo -e "\n复查 tinycudann 内核仍然可用（torchvision 装完后 ABI 是否还完好）"
python3 "${HERE}/probes/tcnn.py" --steps 2 | tail -6 || exit 1
echo -e "\n全部通过。现在重跑:  bash ${HERE}/gates.sh"
