#!/usr/bin/env bash
# Run on the receiving machine: bash scripts/download_work_dirs.sh [directory]
# Re-run the same command after an interruption to resume the download.
set -euo pipefail

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required on both machines. Install it locally first." >&2
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    echo "ssh is required on the receiving machine." >&2
    exit 1
fi

destination=${1:-./downloads}
mkdir -p -- "$destination"
destination=$(cd -- "$destination" && pwd)

echo "Downloading to: $destination/work_dirs.zip"
echo "Enter the server's SSH password when prompted."
echo "If interrupted, run this command again with the same directory to resume."

# Keep partial data; rsync reuses matching blocks when this command is rerun.
# Do not compress an already compressed ZIP or embed credentials in this file.
rsync -t --partial --progress \
    -e 'ssh -p 27068 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=30' \
    root@connect.weste.seetacloud.com:/root/autodl-tmp/work_dirs.zip \
    "$destination/"

echo "Download completed: $destination/work_dirs.zip"
