#!/usr/bin/env python3
"""One-command lpf installer with CUDA auto-detection.

Requires uv (https://docs.astral.sh/uv/):
    curl -LsSf https://astral.sh/uv/install.sh | sh

Usage:
    python install.py            # auto-detect CUDA, install everything
    python install.py --cuda 130 # force CUDA 13.0
    python install.py --cpu      # CPU-only
"""

import argparse
import shutil
import subprocess
import sys


VARIANTS = {
    "126": {"cupy": "cupy-cuda12x", "torch": "torch==2.11.0+cu126", "torch_idx": "cu126"},
    "128": {"cupy": "cupy-cuda12x", "torch": "torch==2.11.0+cu128", "torch_idx": "cu128"},
    "130": {"cupy": "cupy-cuda13x", "torch": "torch==2.11.0+cu130", "torch_idx": "cu130"},
    "132": {"cupy": "cupy-cuda13x", "torch": "torch==2.11.0+cu132", "torch_idx": "cu132"},
}

LPF_WHEEL = (
    "https://github.com/cxinsys/lpf/releases/download/v0.2.0/"
    "lpf-0.2.0+{tag}-py3-none-linux_x86_64.whl"
)


def detect_cuda():
    try:
        out = subprocess.check_output(
            ['nvidia-smi'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'CUDA Version' in line:
                return line.split('CUDA Version:')[-1].strip().split()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None


def pick_variant(cuda_ver):
    key = cuda_ver.replace('.', '')
    if key in VARIANTS:
        return key
    major = cuda_ver.split('.')[0]
    candidates = sorted([k for k in VARIANTS if k.startswith(major)])
    return candidates[-1] if candidates else None


def find_uv():
    uv = shutil.which("uv")
    if uv:
        return uv
    print("uv not found. Installing...")
    subprocess.check_call(
        ["pip", "install", "uv"], stdout=subprocess.DEVNULL)
    return shutil.which("uv") or "uv"


def main():
    parser = argparse.ArgumentParser(description="Install lpf")
    parser.add_argument('--cuda', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--pip', action='store_true',
                        help="Use pip instead of uv")
    args = parser.parse_args()

    use_uv = not args.pip
    if use_uv:
        installer = find_uv()
        run = [installer, "pip", "install"]
    else:
        run = [sys.executable, "-m", "pip", "install"]

    print("=" * 50)
    print("lpf installer")
    print("=" * 50)

    if args.cpu:
        print("\nInstalling CPU-only...")
        subprocess.check_call([*run, "lpf"])
        print("\nDone!")
        return

    if args.cuda:
        key = args.cuda
    else:
        cuda_ver = detect_cuda()
        if cuda_ver is None:
            print("\nNo GPU detected. Use --cpu for CPU-only.")
            sys.exit(1)
        print(f"\nDetected CUDA {cuda_ver}")
        key = pick_variant(cuda_ver)

    if key not in VARIANTS:
        print(f"No wheel for CUDA {key}. Available: {', '.join(sorted(VARIANTS))}")
        sys.exit(1)

    v = VARIANTS[key]
    tag = f"cu{key}"
    torch_index = f"https://download.pytorch.org/whl/{v['torch_idx']}"
    lpf_url = LPF_WHEEL.format(tag=tag)

    print(f"Variant: {tag}\n")

    # Single uv/pip command: all packages at once
    cmd = [
        *run,
        v["cupy"],
        v["torch"],
        lpf_url,
        "--extra-index-url", torch_index,
    ]

    print(f"$ {' '.join(cmd)}\n")
    subprocess.check_call(cmd)

    print(f"\nDone! lpf+{tag} installed.")
    print('Test: python -c "import lpf; print(lpf.__version__)"')


if __name__ == "__main__":
    main()
