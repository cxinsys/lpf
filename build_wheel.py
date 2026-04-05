#!/usr/bin/env python3
"""Build lpf wheel with AOT CUDA kernels.

Usage:
    python build_wheel.py                 # auto-detect GPU, build AOT + wheel
    python build_wheel.py --arch sm_90    # specify GPU arch
    python build_wheel.py --all-arch      # all common archs (80,86,89,90)
    python build_wheel.py --cpu           # CPU-only wheel (no AOT)
    python build_wheel.py --skip-aot      # skip AOT, build wheel with existing binaries
"""

import argparse
import os
import shutil
import subprocess
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.join(_ROOT, "dist")


def _clean():
    """Remove previous build artifacts."""
    for d in ["build", "dist", "lpf.egg-info"]:
        p = os.path.join(_ROOT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"Cleaned: {d}/")


def _build_aot(arch=None, all_arch=False, verbose=False):
    """Run AOT kernel compilation."""
    cmd = [sys.executable, "-m", "lpf.kernels.aot.build"]
    if all_arch:
        cmd.append("--all-arch")
    elif arch:
        cmd.extend(["--arch", arch])
    if verbose:
        cmd.append("-v")

    print("=" * 60)
    print("Step 1: AOT kernel compilation")
    print("=" * 60)
    subprocess.check_call(cmd, cwd=_ROOT)
    print()


def _build_wheel(variant=None):
    """Build the wheel."""
    print("=" * 60)
    print("Step 2: Building wheel")
    print("=" * 60)

    env = os.environ.copy()
    if variant:
        env["LPF_BUILD_VARIANT"] = variant

    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=_ROOT,
        env=env,
    )

    # Print result
    for f in os.listdir(_DIST):
        if f.endswith(".whl"):
            fpath = os.path.join(_DIST, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print()
            print(f"Built: dist/{f}  ({size_mb:.1f} MB)")
            print(f"Install: pip install dist/{f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build lpf wheel")
    parser.add_argument("--arch", type=str, default=None,
                        help="Target SM arch, e.g. sm_90")
    parser.add_argument("--all-arch", action="store_true",
                        help="Build for all common architectures")
    parser.add_argument("--cpu", action="store_true",
                        help="Build CPU-only wheel (skip AOT)")
    parser.add_argument("--skip-aot", action="store_true",
                        help="Skip AOT compilation, use existing binaries")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _clean()

    if args.cpu:
        _build_wheel(variant="cpu")
    else:
        if not args.skip_aot:
            _build_aot(arch=args.arch, all_arch=args.all_arch,
                       verbose=args.verbose)
        _build_wheel()


if __name__ == "__main__":
    main()
