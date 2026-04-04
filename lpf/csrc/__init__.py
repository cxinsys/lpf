"""
lpf.csrc — AOT-compiled CUDA kernels for reaction-diffusion solvers.

Build:   python -m lpf.csrc.build
Load:    from lpf.csrc.loader import AOTKernelLoader
"""

from lpf.csrc.loader import AOTKernelLoader

__all__ = ["AOTKernelLoader"]
