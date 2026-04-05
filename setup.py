import os
import platform
import subprocess

from setuptools import setup, find_packages
from setuptools.command.bdist_wheel import bdist_wheel


# ---------------------------------------------------------------------------
# CUDA version detection
# ---------------------------------------------------------------------------

def _detect_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi.

    Returns a short tag like 'cu132' or None if no CUDA is found.
    """
    try:
        out = subprocess.check_output(
            ['nvcc', '--version'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'release' in line.lower():
                parts = line.split('release')[-1].strip().split(',')[0].strip()
                return 'cu' + parts.replace('.', '')
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        out = subprocess.check_output(
            ['nvidia-smi'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'CUDA Version' in line:
                ver = line.split('CUDA Version:')[-1].strip().split()[0]
                return 'cu' + ver.replace('.', '')
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None


# ---------------------------------------------------------------------------
# Version: 0.2.0+cu132  or  0.2.0+cpu
# ---------------------------------------------------------------------------

_BASE_VERSION = "0.2.0"

_cuda_tag = _detect_cuda_version()
if os.environ.get('LPF_BUILD_VARIANT'):
    _variant = os.environ['LPF_BUILD_VARIANT']
else:
    _variant = _cuda_tag if _cuda_tag else 'cpu'

_version = f"{_BASE_VERSION}+{_variant}"


# ---------------------------------------------------------------------------
# CUDA-specific dependencies (baked into the wheel)
# ---------------------------------------------------------------------------

_CUDA_DEPS = {
    'cu126': ['cupy-cuda12x', 'torch>=2.11'],
    'cu128': ['cupy-cuda12x', 'torch>=2.11'],
    'cu130': ['cupy-cuda13x', 'torch>=2.11'],
    'cu132': ['cupy-cuda13x', 'torch>=2.11'],
}


def _cuda_deps():
    return _CUDA_DEPS.get(_variant, [])


# ---------------------------------------------------------------------------
# Custom bdist_wheel: always platform-specific (contains CUDA kernels)
# ---------------------------------------------------------------------------

def _default_plat_name():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == 'linux':
        return f'manylinux2014_{machine}'
    elif system == 'darwin':
        ver = platform.mac_ver()[0].replace('.', '_')
        return f'macosx_{ver}_{machine}'
    elif system == 'windows':
        return f'win_{machine}'
    return f'{system}_{machine}'


class BdistWheel(bdist_wheel):
    """Always produce a platform-specific wheel (CUDA kernel package)."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        impl = 'py3'
        abi = 'none'
        plat = self.plat_name or _default_plat_name()
        plat = plat.replace('-', '_').replace('.', '_')
        return impl, abi, plat


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name='lpf',
    version=_version,
    description='Ladybird Pattern Formation (LPF)',
    author='Daewon Lee',
    author_email='daewon4you@gmail.com',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'lpf.kernels': ['sources.py'],
        'lpf.kernels.aot': ['*.fatbin', '*.so', '*.dll'],
        'lpf.kernels.csrc': ['*.cu'],
        'lpf.kernels.csrc.include': ['*.cuh'],
    },
    install_requires=[
        'numpy',
        'scipy',
        'pillow',
        'tqdm',
        'pyyaml',
        'xxhash',
    ] + _cuda_deps(),
    extras_require={
        'viz': ['lpips', 'opencv-python', 'torchmetrics'],
        'test': ['pytest'],
    },
    python_requires='>=3.9',
    cmdclass={'bdist_wheel': BdistWheel},
)
