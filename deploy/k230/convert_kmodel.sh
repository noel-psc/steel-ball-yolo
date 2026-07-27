#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 ONNX_PATH CALIBRATION_DIR OUTPUT_KMODEL [SAMPLE_COUNT]" >&2
  exit 2
fi

samples="${4:-100}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/convert_kmodel.py" \
  --onnx "$1" \
  --calibration-dir "$2" \
  --output "$3" \
  --samples "$samples"
