#!/usr/bin/env python3
"""
AOT build script for lpf CUDA kernels.

Compiles kernels.cu → kernels.fatbin using nvcc.
Generates a fat binary containing cubins for detected (or specified)
GPU architectures plus PTX for forward compatibility.

Usage:
    python -m lpf.kernels.aot.build              # auto-detect GPU arch
    python -m lpf.kernels.aot.build --arch sm_90 # explicit arch
    python -m lpf.kernels.aot.build --all-arch   # common archs (80,86,89,90)
"""

import argparse
import os
import shutil
import subprocess
import sys


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC_DIR = os.path.join(_THIS_DIR, "..", "csrc")
_INCLUDE_DIR = os.path.join(_CSRC_DIR, "include")
_SRC_FILE = os.path.join(_CSRC_DIR, "kernels.cu")
_OUT_FILE = os.path.join(_THIS_DIR, "kernels.fatbin")
_LOOP_SRC = os.path.join(_CSRC_DIR, "solver_loop.cu")
_LOOP_OUT = os.path.join(_THIS_DIR, "libsolver.so")

# Common architectures: Ampere (80), Ada Lovelace (86, 89), Blackwell (90)
_COMMON_ARCHS = ["sm_80", "sm_86", "sm_89", "sm_90"]


def find_nvcc():
    """Locate nvcc binary."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError(
            "nvcc not found. Install the CUDA Toolkit and ensure nvcc is on PATH."
        )
    return nvcc


def detect_gpu_arch():
    """Detect compute capability of the first GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            text=True,
        )
        cap = out.strip().split("\n")[0].strip()  # e.g. "9.0"
        major, minor = cap.split(".")
        return f"sm_{major}{minor}"
    except Exception as e:
        raise RuntimeError(f"Could not detect GPU architecture: {e}")


def build(archs=None, verbose=False):
    """Compile kernels.cu into a fat binary.

    Parameters
    ----------
    archs : list[str] or None
        Target architectures, e.g. ["sm_80", "sm_90"].
        If None, auto-detect from the current GPU.
    verbose : bool
        Print the nvcc command before execution.

    Returns
    -------
    str
        Path to the compiled .fatbin file.
    """
    nvcc = find_nvcc()

    if archs is None:
        archs = [detect_gpu_arch()]

    gencode_flags = []
    for arch in archs:
        compute = arch.replace("sm_", "compute_")
        gencode_flags += [
            f"-gencode=arch={compute},code={arch}",
        ]
    # Add PTX for the highest arch (forward compatibility)
    highest = sorted(archs)[-1]
    compute_hi = highest.replace("sm_", "compute_")
    gencode_flags.append(f"-gencode=arch={compute_hi},code={compute_hi}")

    cmd = [
        nvcc,
        "--fatbin",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-I{_INCLUDE_DIR}",
        *gencode_flags,
        _SRC_FILE,
        "-o", _OUT_FILE,
    ]

    if verbose:
        print(" ".join(cmd))

    subprocess.check_call(cmd)
    print(f"Built: {_OUT_FILE}  (archs: {', '.join(archs)})")

    # ---- Build shared library (solver_loop.cu → libsolver.so) ----
    cmd_so = [
        nvcc,
        "-shared", "-Xcompiler", "-fPIC",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-I{_INCLUDE_DIR}",
        *gencode_flags,
        _LOOP_SRC,
        "-o", _LOOP_OUT,
    ]

    if verbose:
        print(" ".join(cmd_so))

    subprocess.check_call(cmd_so)
    print(f"Built: {_LOOP_OUT}  (archs: {', '.join(archs)})")

    return _OUT_FILE, _LOOP_OUT


def main():
    parser = argparse.ArgumentParser(description="Build lpf AOT CUDA kernels")
    parser.add_argument("--arch", type=str, default=None,
                        help="Target SM arch, e.g. sm_90")
    parser.add_argument("--all-arch", action="store_true",
                        help="Build for all common architectures")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.all_arch:
        archs = _COMMON_ARCHS
    elif args.arch:
        archs = [args.arch]
    else:
        archs = None  # auto-detect

    build(archs=archs, verbose=args.verbose)


if __name__ == "__main__":
    main()
