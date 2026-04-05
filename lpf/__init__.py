__version__ = "0.2.0"


def _check_cuda_compat():
    """Warn if CuPy/PyTorch CUDA versions don't match the driver."""
    import warnings
    import subprocess

    # Detect driver CUDA version
    driver_cuda = None
    try:
        out = subprocess.check_output(
            ['nvidia-smi'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'CUDA Version' in line:
                driver_cuda = line.split('CUDA Version:')[-1].strip().split()[0]
                break
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # No GPU, nothing to check

    if driver_cuda is None:
        return

    driver_major = int(driver_cuda.split('.')[0])

    # Check CuPy
    try:
        import cupy
        cupy_cuda = cupy.cuda.runtime.runtimeGetVersion()
        cupy_major = cupy_cuda // 1000
        if cupy_major != driver_major:
            warnings.warn(
                f"CuPy was built for CUDA {cupy_major}.x but the driver "
                f"reports CUDA {driver_cuda}. This may cause errors. "
                f"Install the matching CuPy: pip install cupy-cuda{driver_major}x",
                stacklevel=2,
            )
    except ImportError:
        pass

    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            torch_cuda = torch.version.cuda
            if torch_cuda:
                torch_major = int(torch_cuda.split('.')[0])
                if torch_major != driver_major:
                    warnings.warn(
                        f"PyTorch was built for CUDA {torch_cuda} but the "
                        f"driver reports CUDA {driver_cuda}. "
                        f"Install the matching PyTorch from: "
                        f"https://pytorch.org/get-started/locally/",
                        stacklevel=2,
                    )
    except ImportError:
        pass


_check_cuda_compat()
