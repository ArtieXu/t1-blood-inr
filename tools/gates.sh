#!/usr/bin/env bash
# gates.sh -- 一条命令跑完 5070 接管测试：复制 -> Gate A -> Gate B(U0,U1) -> 打包
#
#   bash tools/gates.sh
#
# 默认行为：把项目从 OneDrive 复制到 WSL 原生文件系统再跑（避开 DrvFs 慢 I/O、
# OneDrive 占位符、以及训练产物被同步上云）。产物一律写到 ext4，不进 OneDrive。
#
# 常用开关：
#   --no-copy            原地跑（仍会把输出放到 ext4）
#   --dest PATH          复制目标，默认 $HOME/<项目名>
#   --out  PATH          产物根目录，默认 $HOME/runs
#   --arms "U0 U1"       要跑的 arm，默认 "U0 U1"；写 "" 表示只做 Gate A
#   --epochs N           冒烟步数，默认 32
#   --force-copy         目标已存在时也重新复制（先备份旧的）
#
# 环境变量：T1_VENV / T1_CUDA_HOME / KB_GRID 会透传给 env.sh 和冒烟脚本。
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "${SRC}")"
DEST="${HOME}/${NAME}"
OUT_ROOT="${T1_OUT_ROOT:-${HOME}/runs}"
ARMS="U0 U1"
EPOCHS=32
DO_COPY=1
FORCE_COPY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-copy)    DO_COPY=0; shift ;;
    --force-copy) FORCE_COPY=1; shift ;;
    --dest)       DEST="$2"; shift 2 ;;
    --out)        OUT_ROOT="$2"; shift 2 ;;
    --arms)       ARMS="$2"; shift 2 ;;
    --epochs)     EPOCHS="$2"; shift 2 ;;
    -h|--help)    sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（-h 看用法）" >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="${OUT_ROOT}/bundle_${STAMP}"
mkdir -p "${BUNDLE}" || { echo "无法创建 ${BUNDLE}" >&2; exit 1; }
MAIN="${BUNDLE}/gates.log"
say() { echo -e "$*" | tee -a "${MAIN}"; }

say "=============================================================="
say " 5070 接管测试  ${STAMP}"
say "=============================================================="
say "源目录:   ${SRC}"
say "产物:     ${BUNDLE}"
grep -qi microsoft /proc/version 2>/dev/null && say "环境:     WSL" || say "环境:     非 WSL Linux（也可以跑）"

# ---------------------------------------------------------------- 1. 复制
PROJ="${SRC}"
if [ "${DO_COPY}" -eq 1 ] && [ "$(cd "${SRC}" && pwd -P)" = "$(cd "$(dirname "${DEST}")" 2>/dev/null && pwd -P || echo /nonexistent)/$(basename "${DEST}")" ]; then
  say "\n源目录就是目标目录，跳过复制：${SRC}"
  DO_COPY=0
fi
if [ "${DO_COPY}" -eq 1 ]; then
  case "${SRC}" in
    /mnt/[a-z]/*) say "\n源在 Windows 盘上，按计划复制到 ext4。" ;;
    *)            say "\n源已经不在 /mnt/ 下，仍按要求复制到 ${DEST}。" ;;
  esac
  if [ -e "${DEST}" ] && [ "${FORCE_COPY}" -eq 0 ]; then
    say "目标已存在，复用其中的数据（要整体重新复制加 --force-copy）：${DEST}"
    # 但 tools 每次都从源目录刷新：它是工具，很小，而且新交付的脚本
    # 只会出现在源目录里。不刷新的话会一直跑到旧快照。
    if [ -d "${SRC}/tools" ] && \
       [ "$(cd "${SRC}" && pwd -P)" != "$(cd "${DEST}" && pwd -P)" ]; then
      mkdir -p "${DEST}/tools"
      if cp -a "${SRC}/tools/." "${DEST}/tools/" 2>>"${MAIN}"; then
        say "  tools 已从源目录刷新到最新"
      else
        say "  警告：tools 刷新失败，将使用副本里的旧版本"
      fi
    fi
  else
    if [ -e "${DEST}" ]; then
      BAK="${DEST}.bak_${STAMP}"; say "备份旧目录 -> ${BAK}"; mv "${DEST}" "${BAK}" || exit 1
    fi
    say "复制中…（OneDrive 按需文件会在这一步被拉下来，可能较慢）"
    if ! cp -r "${SRC}" "${DEST}" 2>>"${MAIN}"; then
      say "复制失败。最常见原因是 OneDrive 占位符没能下载。"
      say "请在资源管理器里右键该文件夹 ->「始终保留在此设备上」，等同步完再重跑。"
      exit 1
    fi
    say "复制完成：$(du -sh "${DEST}" 2>/dev/null | cut -f1)"
    SRC_RECORD="${SRC}"        # 记住源，以后每次运行自动刷新工具包
  fi
  PROJ="${DEST}"
else
  say "\n--no-copy：原地跑 ${PROJ}（产物仍写到 ${OUT_ROOT}）"
fi
KIT="${PROJ}/tools"
[ -d "${KIT}" ] || { say "找不到 ${KIT}"; exit 1; }

# 工具包自动刷新：首次复制时会把源目录记进 .upstream，之后每次运行都从那里
# 把 tools 更新到最新。新交付的脚本只会出现在源目录（OneDrive）里，
# 不刷新的话会一直跑到旧快照。
# --remove-destination 先 unlink 再写，这样正在运行的本脚本保住自己的 inode，
# 覆盖不会让 bash 读到半新半旧的内容（已实测验证）。
if [ -n "${SRC_RECORD:-}" ]; then echo "${SRC_RECORD}" > "${KIT}/.upstream"; fi
if [ -f "${KIT}/.upstream" ]; then
  KIT_SRC="$(head -1 "${KIT}/.upstream")"
  if [ -d "${KIT_SRC}/tools" ] &&
     [ "$(cd "${KIT_SRC}" && pwd -P)" != "$(cd "${PROJ}" && pwd -P)" ]; then
    if cp -a --remove-destination "${KIT_SRC}/tools/." "${KIT}/" 2>>"${MAIN}"; then
      echo "${KIT_SRC}" > "${KIT}/.upstream"
      sed -i 's/\r$//' "${KIT}"/*.sh 2>/dev/null
      say "工具包已从 ${KIT_SRC} 刷新到最新"
    else
      say "警告：工具包刷新失败，使用副本里的现有版本"
    fi
  fi
fi

# ------------------------------------------------------- 2. 修 CRLF + 环境
say "\n[1/4] 规范换行符（OneDrive/Windows 编辑器会带进 CRLF）"
sed -i 's/\r$//' "${KIT}"/*.sh "${KIT}"/*.py 2>/dev/null
say "      done"

say "\n[2/4] 激活环境"
# 注意：不能写成 `source ... | tee`，管道会开子 shell，export 出来的
# VIRTUAL_ENV / CUDA_HOME 全部丢失。先重定向到文件，再回显。
# shellcheck disable=SC1090
source "${KIT}/env.sh" > "${BUNDLE}/env_activate.log" 2>&1
cat "${BUNDLE}/env_activate.log" | tee -a "${MAIN}"
if [ -z "${VIRTUAL_ENV:-}" ] || [ -z "${CUDA_HOME:-}" ]; then
  say "\n环境没起来。检查这两项后重试："
  say "  T1_VENV      现在找的是 \${HOME}/.venvs/t1_5070"
  say "  T1_CUDA_HOME 现在找的是 /usr/local/cuda-12.8"
  say "例：T1_VENV=/home/t1user/.venvs/t1_5070 bash ${BASH_SOURCE[0]} $*"
  exit 1
fi
export T1_OUT_ROOT="${OUT_ROOT}"

# ---------------------------------------------------------------- 3. Gate A
say "\n[3/4] Gate A：环境复核（含 tcnn 内核实跑）"
bash "${KIT}/doctor.sh" 2>&1 | tee -a "${MAIN}"
GATE_A=${PIPESTATUS[0]}
LATEST_A="$(find "${OUT_ROOT}" -maxdepth 1 -name 'doctor_*' -type d | sort | tail -1)"
[ -n "${LATEST_A}" ] && cp -r "${LATEST_A}" "${BUNDLE}/" 2>/dev/null
if [ "${GATE_A}" -ne 0 ]; then
  say "\nGate A 未通过 —— 不继续跑 Gate B。"
  say "证据包：${BUNDLE}"
  exit 1
fi
say "Gate A 通过。"

# ---------------------------------------------------------------- 4. Gate B
say "\n[4/4] Gate B：${EPOCHS} 步冒烟，arms = ${ARMS:-<跳过>}"
GATE_B_FAIL=0
for arm in ${ARMS}; do
  say "\n---------- arm ${arm} ----------"
  ARM="${arm}" EPOCHS="${EPOCHS}" bash "${KIT}/smoke.sh" 2>&1 | tee -a "${MAIN}"
  rc=${PIPESTATUS[0]}
  [ "${rc}" -ne 0 ] && { GATE_B_FAIL=1; say "arm ${arm} 失败（退出码 ${rc}）"; }
  LATEST_B="$(find "${OUT_ROOT}" -maxdepth 1 -name "smoke_${arm}_*" -type d | sort | tail -1)"
  [ -n "${LATEST_B}" ] && cp -r "${LATEST_B}" "${BUNDLE}/" 2>/dev/null
done

# ------------------------------------------------------------- 5. 打包摘要
SUM="${BUNDLE}/SUMMARY.txt"
{
  echo "5070 接管测试摘要  ${STAMP}"
  echo "项目: ${PROJ}"
  echo "Gate A: $([ "${GATE_A}" -eq 0 ] && echo PASS || echo FAIL)"
  echo "Gate B: $([ "${GATE_B_FAIL}" -eq 0 ] && echo PASS || echo FAIL)   arms=${ARMS} epochs=${EPOCHS}"
  echo
  echo "---- runtime signature ----"
  for f in "${BUNDLE}"/doctor_*/sysinfo.json; do
    [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open('$f'))
print(' gpu       ', d.get('gpu',{}).get('name'), d.get('gpu',{}).get('capability'), f\"{d.get('gpu',{}).get('total_memory_gib','?')} GiB\")
print(' driver    ', (d.get('nvidia_smi') or '').split(',')[1].strip() if d.get('nvidia_smi') else '?')
print(' torch     ', d.get('torch',{}).get('version'), 'cuda', d.get('torch',{}).get('cuda_runtime'))
for k in ('torchvision','torchkbnufft','tinycudann','numpy','scipy','h5py','skimage'):
    print(f' {k:10s}', d.get('packages',{}).get(k))
" 2>/dev/null
  done
  echo
  echo "---- Gate B 验收 ----"
  for f in "${BUNDLE}"/smoke_*/acceptance.txt; do
    [ -f "$f" ] && { echo "== $(basename "$(dirname "$f")") =="; cat "$f"; echo; }
  done
  echo "---- 系统内存峰值 ----"
  for f in "${BUNDLE}"/smoke_*/host_resources.txt; do
    [ -f "$f" ] && { echo -n "$(basename "$(dirname "$f")"): "; grep -i 'Maximum resident' "$f" || echo '(未记录)"'; }
  done
} > "${SUM}" 2>&1

# 把纯文本产物回写到源目录（OneDrive），这样不用手动粘贴也能被读到。
# 只带 txt/log/json/csv —— .mat/.pt/.png 留在 ext4，绝不往 OneDrive 上传。
if [ -f "${KIT}/.upstream" ]; then
  REPORT_SRC="$(head -1 "${KIT}/.upstream")"
  if [ -d "${REPORT_SRC}" ]; then
    REPORT_DIR="${REPORT_SRC}/_gate_reports"
    mkdir -p "${REPORT_DIR}" 2>/dev/null
    if ( cd "$(dirname "${BUNDLE}")" && find "$(basename "${BUNDLE}")" \
           \( -name '*.txt' -o -name '*.log' -o -name '*.json' -o -name '*.csv' \) \
           -exec cp --parents {} "${REPORT_DIR}/" \; ) 2>>"${MAIN}"; then
      say "\n文本报告已回写到源目录（体积很小，不含 .mat/.pt/.png）："
      say "  ${REPORT_DIR}/$(basename "${BUNDLE}")"
    fi
  fi
fi

say "\n=============================================================="
say " Gate A: $([ "${GATE_A}" -eq 0 ] && echo PASS || echo FAIL)"
say " Gate B: $([ "${GATE_B_FAIL}" -eq 0 ] && echo PASS || echo FAIL)"
say "=============================================================="
say "证据包:   ${BUNDLE}"
say "贴回来这一个文件即可: ${SUM}"
say ""
say "  cat ${SUM}"
exit $(( GATE_A != 0 || GATE_B_FAIL != 0 ))
