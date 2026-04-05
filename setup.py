from setuptools import setup, find_packages

setup(
    name='lpf',
    version="0.2.0",
    description='Ladybird Pattern Formation (LPF)',
    author='Daewon Lee',
    author_email='daewon4you@gmail.com',
    packages=find_packages(),
    package_data={
        'lpf.kernels.aot': ['*.fatbin', '*.so'],
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
    ],
    extras_require={
        'viz': ['lpips', 'opencv-python', 'torchmetrics'],
        'test': ['pytest'],
    },
    python_requires='>=3.9',
)
