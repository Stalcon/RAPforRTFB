#!/usr/bin/env bash
set -euo pipefail

detect_backend() {
    local device vendor device_class amd_found=false

    if command -v nvidia-smi >/dev/null 2>&1 &&
        nvidia-smi -L 2>/dev/null | grep '^GPU ' >/dev/null; then
        echo cuda
        return
    fi
    if command -v rocminfo >/dev/null 2>&1 &&
        rocminfo 2>/dev/null | grep -E '^[[:space:]]*Name:[[:space:]]+gfx[[:xdigit:]]+' >/dev/null; then
        echo rocm
        return
    fi

    # PCI enumeration still works on Linux before the GPU driver is installed.
    for device in /sys/bus/pci/devices/*; do
        [[ -r "$device/class" && -r "$device/vendor" ]] || continue
        read -r device_class < "$device/class"
        [[ "$device_class" == 0x03* ]] || continue
        read -r vendor < "$device/vendor"
        case "$vendor" in
            0x10de) echo cuda; return ;;
            0x1002) amd_found=true ;;
        esac
    done

    if [[ "$amd_found" == true ]]; then
        echo rocm
    else
        echo cpu
    fi
}

ENV_NAME="${1:-rap}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${BACKEND:-auto}"
if [[ "$BACKEND" == auto ]]; then
    BACKEND="$(detect_backend)"
fi

case "$BACKEND" in
    rocm)
        if [ "$(uname -s)" != Linux ]; then
            echo 'Use a supported Linux/ROCm installation for AMD acceleration.' >&2
            exit 1
        fi
        TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
        ;;
    cuda) TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" ;;
    cpu) TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" ;;
    *) echo 'Set BACKEND to auto, cpu, rocm, or cuda.' >&2; exit 1 ;;
esac

if command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo 'Install conda or mamba first.' >&2
    exit 1
fi
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA" create -n "$ENV_NAME" python=3.11 -y
fi

echo "Using $BACKEND for UMA."
torch_install=(python -m pip install 'torch==2.8.0' --index-url "$TORCH_INDEX")
"$CONDA" run -n "$ENV_NAME" "${torch_install[@]}"
installed_backend="$("$CONDA" run -n "$ENV_NAME" python -c '
import torch

if torch.version.hip:
    print("rocm")
elif torch.version.cuda:
    print("cuda")
else:
    print("cpu")
')"
# The public version pin can match an existing wheel for a different backend.
if [[ "$installed_backend" != "$BACKEND" ]]; then
    "$CONDA" run -n "$ENV_NAME" "${torch_install[@]}" --force-reinstall
fi

"$CONDA" run -n "$ENV_NAME" python -m pip install -e "$ROOT[uma]"
dft_target="${DFT_BACKEND:-auto}"
if [[ "$dft_target" == auto ]]; then
    dft_target="$BACKEND"
fi
case "$dft_target" in
    cuda)
        "$CONDA" run -n "$ENV_NAME" python -m pip install gpu4pyscf-cuda12x cutensor-cu12
        ;;
    rocm)
        if ! "$CONDA" run -n "$ENV_NAME" python -c 'import sys, cupy; sys.exit(not (cupy.cuda.runtime.is_hip and cupy.cuda.runtime.getDeviceCount() > 0))'; then
            echo "Install a working ROCm build of CuPy in $ENV_NAME, then rerun setup. See README.md." >&2
            exit 1
        fi
        if ! DFT_BACKEND=rocm "$CONDA" run -n "$ENV_NAME" python -c 'from core.backend import get_dft_module; get_dft_module()'; then
            hip_source="git+https://github.com/pyscf/gpu4pyscf.git@6cf69ba3bd9f530039e7ca23e38ec062184fad81"
            GPU4PYSCF_BACKEND=hip "$CONDA" run -n "$ENV_NAME" python -m pip install "$hip_source"
        fi
        ;;
    cpu) ;;
    *) echo 'Set DFT_BACKEND to auto, cpu, rocm, or cuda.' >&2; exit 1 ;;
esac
BACKEND="$BACKEND" "$CONDA" run -n "$ENV_NAME" python "$ROOT/scripts/check_env.py"
echo "Activate with: conda activate $ENV_NAME"
echo 'UMA needs approved Hugging Face model access. Authenticate with hf auth login.'
