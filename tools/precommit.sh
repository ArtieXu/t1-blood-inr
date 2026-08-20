#!/usr/bin/env bash
# precommit.sh -- 第一步：原地建 git 仓库，但在 commit 之前先做安全审计。
#
#   bash precommit.sh              # 只审计，不提交（默认，安全）
#   bash precommit.sh --commit     # 审计通过后才真正 commit
#
# 不移动任何文件，不改任何代码。只做三件事：
#   1. 写 .gitignore
#   2. git init + git add（只是暂存，还没提交）
#   3. 审计「将要被提交的内容」—— 大文件、受试者数据、凭据、notebook 内嵌图像
#
# 审计不通过就停下，绝不自动提交。
set -uo pipefail
REPO="${1:-$PWD}"; [ "${REPO}" = "--commit" ] && REPO="$PWD"
DO_COMMIT=0; for a in "$@"; do [ "$a" = "--commit" ] && DO_COMMIT=1; done
cd "${REPO}" || exit 1
echo "仓库目录: $(pwd)"
command -v git >/dev/null || { echo "没装 git: sudo apt install git" >&2; exit 1; }

# ---------------------------------------------------------------- 1. .gitignore
if [ -f .gitignore ]; then
  echo "已存在 .gitignore，备份为 .gitignore.bak 后覆盖"
  cp .gitignore .gitignore.bak
fi
cat > .gitignore <<'EOF'
# =====================================================================
#  T1 blood INR —— 仓库只放「代码 + 配置 + 文档 + 文本报告」
#
#  原则：二进制和图像一律默认拒绝（default-deny），文本一律放行。
#  重建出来的图像属于受试者数据，绝不能进入任何远程仓库。
#  确实需要提交某个被挡住的文件时，显式 git add -f <文件>。
# =====================================================================

# ---- 数据与模型权重（受试者数据 / 大文件）----
*.mat
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.h5
*.hdf5
*.nii
*.nii.gz
*.dcm
*.raw

# ---- 图像产物（重建影像 = 受试者数据）----
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.tif
*.tiff
*.svg
*.pdf
*.mp4

# ---- 运行产物目录（大文件的老家）----
# 训练产物目录（仓库根目录下的）。注意不要写成 **/log/ ——
# 那会把 _gate_reports/*/smoke_*/log/*/loss.csv 也挡掉，而那是给 AI 看结果的通道。
# 二进制本来就被上面的扩展名规则拦住了，这里只需挡根目录的 log/。
/log/
runs/
runs/
out/
wandb/

# ---- 凭据 ----
.env
.env.*
*.key
*.pem
*credential*
*secret*
.netrc

# ---- Python ----
__pycache__/
*.py[cod]
*.so
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
*.egg-info/

# ---- 编辑器 / 系统 ----
.vscode/
.idea/
.DS_Store
Thumbs.db
desktop.ini
EOF
echo "✓ .gitignore 已写入"

# ---------------------------------------------------------------- 2. init + add
[ -d .git ] || { git init -q; echo "✓ git init"; }
git add -A 2>/dev/null
echo "✓ 已暂存（尚未提交）"

# ---------------------------------------------------------------- 3. 安全审计
echo; echo "══════════════════ 安全审计 ══════════════════"
FAIL=0
N=$(git diff --cached --name-only | wc -l)
SZ=$(git diff --cached --name-only | xargs -r du -ch 2>/dev/null | tail -1 | cut -f1)
echo "将要提交: ${N} 个文件, 合计 ${SZ:-?}"

echo; echo "[1/5] 大文件检查（阈值 1 MB）"
BIG=$(git diff --cached --name-only | while read -r f; do
        [ -f "$f" ] && s=$(stat -c%s "$f") && [ "$s" -gt 1048576 ] &&
          awk -v b="$s" -v n="$f" 'BEGIN{printf "%8.1f MB  %s\n", b/1048576, n}'; done)
if [ -n "${BIG}" ]; then echo "${BIG}"; echo "  ✗ 有大文件被暂存 —— .gitignore 没挡住"; FAIL=1
else echo "  ✓ 没有超过 1 MB 的文件"; fi

echo; echo "[2/5] 受试者数据 / 二进制扩展名"
BAD=$(git diff --cached --name-only | grep -iE '\.(mat|pt|pth|ckpt|npy|npz|h5|hdf5|nii|dcm|png|jpg|jpeg|tif|tiff|pdf)$')
if [ -n "${BAD}" ]; then echo "${BAD}" | sed 's/^/    /'; echo "  ✗ 这些不该进仓库"; FAIL=1
else echo "  ✓ 没有数据或影像文件"; fi

echo; echo "[3/5] notebook 是否带已执行输出或内嵌图像"
# 用数组而不是字符串，文件名带空格也不会散架
mapfile -t NB_ARR < <(git diff --cached --name-only | grep '\.ipynb$')
if [ "${#NB_ARR[@]}" -gt 0 ]; then
  NBPY="$(mktemp)"
  cat > "${NBPY}" <<'PYEOF'
import json, os, sys
bad = 0
for f in sys.argv[1:]:
    try:
        nb = json.load(open(f))
    except Exception as e:
        print(f"    ? {os.path.basename(f):50s} 读不出来: {e}"); bad = 1; continue
    cells = nb.get('cells', [])
    outs = sum(len(c.get('outputs', [])) for c in cells)
    imgs = sum(1 for c in cells for o in c.get('outputs', [])
               for k in o.get('data', {}) if str(k).startswith('image/'))
    if imgs: bad = 1
    mark = "!! 内嵌图像，不能提交" if imgs else ("有文本输出" if outs else "干净")
    print(f"    {'X' if imgs else 'o'} {os.path.basename(f):50s} 输出={outs} 图像={imgs}  {mark}")
sys.exit(1 if bad else 0)
PYEOF
  # 脚本走文件、notebook 列表走 argv —— 只有一个 stdin，不会互相覆盖
  if python3 "${NBPY}" "${NB_ARR[@]}"; then
    echo "  ✓ notebook 都不含内嵌图像"
  else
    echo "  ✗ 有 notebook 带内嵌图像（或读不出来）—— 重建影像属于受试者数据"
    echo "    先清输出: jupyter nbconvert --clear-output --inplace <文件>"
    FAIL=1
  fi
  rm -f "${NBPY}"
else
  echo "  ✓ 没有 notebook 被暂存"
fi

echo; echo "[4/5] 凭据扫描"
CRED=$(git diff --cached --name-only | xargs -r grep -IlE \
  'sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{30,}|WANDB_API_KEY *= *["'"'"'][^"'"'"']{10,}|xox[baprs]-[A-Za-z0-9-]{10,}' 2>/dev/null)
if [ -n "${CRED}" ]; then echo "${CRED}" | sed 's/^/    ✗ /'; echo "  ✗ 疑似凭据，逐个人工确认"; FAIL=1
else echo "  ✓ 未发现 API key / token 形态的字符串"
     echo "    （注意：40 位十六进制的 SHA-256 和 commit hash 不算凭据，已排除）"; fi

echo; echo "[5/5] 实际会提交的文件清单"
git diff --cached --name-only | sed 's/^/    /'

echo; echo "══════════════════════════════════════════════"
if [ "${FAIL}" -ne 0 ]; then
  echo "审计未通过。先修上面的问题，再重跑本脚本。"
  echo "（当前只是暂存，什么都还没提交。撤销: git reset）"
  exit 1
fi
echo "审计通过。"
if [ "${DO_COMMIT}" -eq 1 ]; then
  git -c user.name="${GIT_NAME:-$(whoami)}" -c user.email="${GIT_EMAIL:-$(whoami)@localhost}" \
      commit -q -m "Initial commit: T1 blood INR (code, docs, tooling; data excluded)"
  echo "✓ 已提交: $(git rev-parse --short HEAD)"
  echo
  echo "下一步 —— 推到 GitHub："
  echo "  1) 在 github.com 新建一个【私有】空仓库，名字 t1-blood-inr，不要勾 README"
  echo "  2) git remote add origin git@github.com:<你的用户名>/t1-blood-inr.git"
  echo "     （或 https://github.com/<你的用户名>/t1-blood-inr.git）"
  echo "  3) git branch -M main && git push -u origin main"
else
  echo "这是预演。确认上面的清单没问题后，加 --commit 重跑："
  echo "  bash $0 --commit"
fi
