#!/usr/bin/env bash
set -euo pipefail

USAGE="Usage: $0 <local_path> <remote_dir>\n\n  <local_path>  本地文件或目录\n  <remote_dir>  OneDrive 上的目标目录（相对于根目录，不以 / 开头）"

if [ $# -ne 2 ]; then
  echo -e "$USAGE"
  exit 1
fi

LOCAL_PATH="$1"
REMOTE_DIR="$2"
REMOTE_NAME="onedrive"   # 上面 config 时设置的 name

# 检查 rclone 是否可用
if ! command -v rclone &>/dev/null; then
  echo "Error: rclone 未安装，请先安装并配置好 onedrive remote。"
  exit 2
fi

echo "正在上传 ${LOCAL_PATH} → ${REMOTE_NAME}:${REMOTE_DIR} ..."
rclone copy "$LOCAL_PATH" "${REMOTE_NAME}:${REMOTE_DIR}" \
  --transfers=4 \
  --checkers=8 \
  --progress

echo "上传完成！"
