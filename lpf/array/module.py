import warnings
import numpy as np


def parse_device(device):
    # jax:gpu:0
    # jax:cuda:
    # cupy:cuda:0
    if device is None:
        return "numpy", "cpu", 0

    device = device.lower()

    _backend = None
    _device = None
    _device_id = 0

    if ":" in device:
        options = device.split(":")
        if len(options) == 2:
            if str.isnumeric(options[1]):
                _device_id = int(options[1])

                # The default device is GPU if the device is not defined.
                if options[0] in ["cuda", "cupy"]:
                    _backend = "cupy"
                    _device = "gpu"
                elif options[0] == "torch":
                    _backend = "torch"
                    _backend = "gpu"
                elif options[0] == "jax":
                    _backend = "jax"
                    _device = "gpu"

            elif isinstance(options[-1], str):
                _backend, _device = options
                _device_id = 0

        elif len(options) == 3:
            _backend, _device, _device_id = options

            if _backend not in ["numpy", "cupy", "torch", "jax"]:
                raise ValueError("backend should be one of 'numpy', 'cupy', 'torch', or "\
                                 "'jax', not %s" % (_backend))
        else:
            raise RuntimeError("Illegal device:", device)

        _device_id = int(_device_id)

    else:
        if device in ["cpu", "numpy"]:
            _backend = "numpy"
            _device = "cpu"
            _device_id = 0
        elif device == "cupy":
            _backend = "cupy"
            _device = "gpu"
            _device_id = 0
        elif device == "torch":
            _backend = "torch"
            _device = "gpu"
            _device_id = 0
        elif device == "jax":
            _backend = "jax"
            _device = "gpu"
            _device_id = 0
        else:
            raise ValueError("Illegal device:", device)

    if _device not in ["cpu", "gpu", "tpu", "cuda"]:
        raise ValueError("device should be one of 'cpu', " \
                         "'gpu', or 'cuda', not %s" % (_device))

    if _backend == None and _device in ["gpu", "cuda"]:
        _backend = "cupy"  # The default backend for gpu is cupy.
    elif _backend == None and _device == "cpu":
        _backend = "numpy"

    return _backend, _device, _device_id


def get_array_module(device):
    _backend, _device, _device_id = parse_device(device)

    if _backend == "cupy" and _device in ["gpu", "cuda"]:
        return CupyModule(_device, _device_id)
    elif _backend == "torch":
        if _device in ["gpu", "cuda"]:
            return TorchModule("cuda", _device_id)
        else:
            return TorchModule("cpu", 0)
    elif _backend == "jax":
        if _device in ["gpu", "cuda"]:
            return JaxModule("gpu", _device_id)
        else:
            return JaxModule("cpu", 0)
    else:
        return NumpyModule(_device, _device_id)


class ArrayModule:
    def __init__(self, device, device_id):
        self._device = device
        self._device_id = device_id

    def __enter__(self):
        return

    def __exit__(self, *args, **kwargs):
        return

    @property
    def device(self):
        return self._device

    @property
    def device_id(self):
        return self._device_id


class NumpyModule(ArrayModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

    def any(self, *args, **kwargs):
        return np.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return np.ones(*args, **kwargs)

    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def abs(self, *args, **kwargs):
        return np.abs(*args, **kwargs)

    def get(self, arr):
        return arr

    def set(self, arr, ind, val):
        arr[ind] = val
        return arr

    def is_array(self, obj):
        return isinstance(obj, (np.ndarray, np.generic))

    def isnan(self, *args, **kwargs):
        return np.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        return np.isinf(*args, **kwargs)


class CupyModule(NumpyModule):

    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)
        
        import cupy as cp        
        self._module = cp
        self._device = self._module.cuda.Device()
        self._device.id = self._device_id
        self._device.use()

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def any(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.ones(*args, **kwargs)

    def array(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.array(*args, **kwargs)

    def abs(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.abs(*args, **kwargs)

    def get(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()

        return arr

    def is_array(self, obj):
        return isinstance(obj, (self._module.ndarray, self._module.generic))

    def isnan(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        with self._module.cuda.Device(self.device_id):
            return self._module.isinf(*args, **kwargs)

    def clear_memory(self):
        import gc
        gc.collect()
        with self._module.cuda.Device(self.device_id) as dev:
            self._module.get_default_memory_pool().free_all_blocks()
            self._module.get_default_pinned_memory_pool().free_all_blocks()
            self._module.dev.synchronize()



class TorchModule(ArrayModule):

    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

        import torch
        self._module = torch

        if device is None:
            self._device = torch.device('cpu')
        elif device.startswith('cuda'):
            if device_id is not None:
                self._device = torch.device('cuda', device_id)
            else:
                self._device = torch.device('cuda')
        else:
            self._device = torch.device(device)

        self._NP_TO_TORCH_DTYPE = {
            np.dtype(bool):         torch.bool,
            np.dtype(np.bool_):     torch.bool,

            np.dtype(np.int8):     torch.int8,
            np.dtype(np.int16):    torch.int16,
            np.dtype(np.int32):    torch.int32,
            np.dtype(np.int64):    torch.int64,

            np.dtype(np.uint8):     torch.uint8,
            np.dtype(np.uint16):    torch.int16,
            np.dtype(np.uint32):    torch.int32,
            np.dtype(np.uint64):    torch.int64,

            np.dtype(np.float16):   torch.float16,
            np.dtype(np.float32):   torch.float32,
            np.dtype(np.float64):   torch.float64,
            np.dtype(np.longdouble): torch.float64,

            np.dtype(np.complex64):  torch.complex64,
            np.dtype(np.complex128): torch.complex128
        }

    def _np_dtype_to_torch_dtype(self, dt):
        if isinstance(dt, self._module.dtype):
            return dt

        try:
            dt = np.dtype(dt)
        except TypeError:
            raise ValueError(f"Cannot convert {dt} to a NumPy dtype.")

        torch_dtype = self._NP_TO_TORCH_DTYPE.get(dt, None)
        if torch_dtype is None:
            raise ValueError(f"No corresponding torch dtype for NumPy dtype {dt}.")
        return torch_dtype

    def any(self, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = self.device

        if 'dtype' in kwargs:
            kwargs['dtype'] = self._np_dtype_to_torch_dtype(kwargs['dtype'])

        return self._module.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = self.device

        if 'dtype' in kwargs:
            kwargs['dtype'] = self._np_dtype_to_torch_dtype(kwargs['dtype'])

        return self._module.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = self.device

        if 'dtype' in kwargs:
            kwargs['dtype'] = self._np_dtype_to_torch_dtype(kwargs['dtype'])

        return self._module.ones(*args, **kwargs)

    def array(self, data, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = self.device

        if 'dtype' in kwargs:
            kwargs['dtype'] = self._np_dtype_to_torch_dtype(kwargs['dtype'])

        return self._module.tensor(data, *args, **kwargs)

    def abs(self, *args, **kwargs):
        return self._module.abs(*args, **kwargs)

    def get(self, arr):
        return arr.cpu().detach().numpy()

    def set(self, arr, ind, val):
        arr[ind] = val
        return arr

    def is_array(self, obj):
        return isinstance(obj, self._module.Tensor)

    def isnan(self, *args, **kwargs):
        return self._module.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        return self._module.isinf(*args, **kwargs)

    def clear_memory(self):
        import gc
        gc.collect()

        self._module.cuda.set_device(self.device_id)
        self._module.cuda.empty_cache()
        self._module.cuda.synchronize()


class JaxModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

        
        import os
    
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
        
        import jax
        from jax import device_put
        import jax.numpy as jnp
                
        self._module = jnp
        self._device_put = device_put

        for jax_device in jax.devices(device):
            if jax_device.id == device_id:
                self._device = jax_device
                break
        # end of for

    def any(self, *args, **kwargs):
        return self._module.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return self._device_put(self._module.zeros(*args, **kwargs),
                                self._device)

    def ones(self, *args, **kwargs):
        return self._device_put(self._module.ones(*args, **kwargs),
                                self._device)

    def array(self, *args, **kwargs):
        return self._device_put(self._module.array(*args, **kwargs),
                                self._device)

    def abs(self, *args, **kwargs):
        return self._module.abs(*args, **kwargs)

    def get(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()

        return arr

    def set(self, arr, ind, val):
        return arr.at[ind].set(val)

    def is_array(self, obj):
        return isinstance(obj, (self._module.ndarray, self._module.generic))

    def isnan(self, *args, **kwargs):
        return self._module.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        return self._module.isinf(*args, **kwargs)

    def clear_memory(self):
        jnp = self._module.numpy

        self._module.clear_backends()
        self._module.numpy.array([], dtype=jnp.float32).device_put(self.device)

        import gc
        gc.collect()