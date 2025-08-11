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

        # Platform dependent pointer sized int
        _INTP_TARGET = self._module.int64 if np.dtype(np.intp).itemsize == 8 else self._module.int32

        # NumPy dtype -> torch.dtype mapping
        # Note: for unsupported unsigned integer types we map to a safe signed type
        self._NP_TO_TORCH_DTYPE = {
            np.dtype(bool):           self._module.bool,
            np.dtype(np.bool_):       self._module.bool,

            np.dtype(np.int8):        self._module.int8,
            np.dtype(np.int16):       self._module.int16,
            np.dtype(np.int32):       self._module.int32,
            np.dtype(np.int64):       self._module.int64,
            np.dtype(np.intp):        _INTP_TARGET,

            np.dtype(np.uint8):       self._module.uint8,
            np.dtype(np.uint16):      self._module.int32,   # no uint16 in torch
            np.dtype(np.uint32):      self._module.int64,   # no uint32 in torch
            np.dtype(np.uint64):      self._module.int64,   # no uint64 in torch

            np.dtype(np.float16):     self._module.float16,
            np.dtype(np.float32):     self._module.float32,
            np.dtype(np.float64):     self._module.float64,
            np.dtype(np.longdouble):  self._module.float64, # treat as float64 for compatibility

            np.dtype(np.complex64):   getattr(self._module, "complex64"),
            np.dtype(np.complex128):  getattr(self._module, "complex128"),
        }

        # torch.dtype -> NumPy dtype mapping
        # Used when user asks for a torch dtype and we need a NumPy dtype for pre casting
        self._TORCH_TO_NP_DTYPE = {
            self._module.bool:        np.bool_,
            self._module.uint8:       np.uint8,
            self._module.int8:        np.int8,
            self._module.int16:       np.int16,
            self._module.int32:       np.int32,
            self._module.int64:       np.int64,
            self._module.float16:     np.float16,
            self._module.float32:     np.float32,
            self._module.float64:     np.float64,
            getattr(self._module, "complex64"):   np.complex64,
            getattr(self._module, "complex128"):  np.complex128,
        }

    def _convert_numpy_dtype_for_torch(self, npdt: np.dtype) -> np.dtype:
        """
        Convert a NumPy dtype into a dtype that is compatible with torch.tensor().

        Rules:
        - uint16 -> int32   (no uint16 in torch)
        - uint32 -> int64   (no uint32 in torch)
        - uint64 -> keep as uint64 for now, handle overflow separately
        - float128 / longdouble -> float64
        - others -> unchanged
        """
        if npdt == np.dtype(np.uint16):
            return np.dtype(np.int32)
        if npdt == np.dtype(np.uint32):
            return np.dtype(np.int64)
        if npdt == np.dtype(np.uint64):
            # Keep uint64 for now and decide at array casting time
            return npdt
        if str(npdt) in ("float128", "longdouble"):
            return np.dtype(np.float64)
        return npdt

    def _np_dtype_to_torch_dtype(self, data=None, dtype_hint=None):
        """
        Decide the appropriate torch.dtype for given data or dtype_hint, and cast
        NumPy arrays only when necessary for torch compatibility.

        Parameters
        ----------
        data : Any
            Optionally a NumPy array to be cast if needed.
        dtype_hint : Any
            Optional dtype hint (torch.dtype, np.dtype, string, or None).

        Returns
        -------
        data_out : Any
            Possibly cast NumPy array (if `data` was an ndarray and casting was required),
            otherwise unchanged.
        torch_dtype : Optional[torch.dtype]
            The resolved torch dtype, or None if left for torch to infer.
        """
        def _cast_ndarray_minimal(arr: np.ndarray, target_np_dtype: np.dtype) -> np.ndarray:
            """Cast only if dtypes differ. Uses copy=False to avoid extra allocations."""
            if arr.dtype == target_np_dtype:
                return arr
            return arr.astype(target_np_dtype, copy=False)

        # Case 1: dtype_hint is a torch.dtype. Respect it and pre cast if necessary.
        if isinstance(dtype_hint, self._module.dtype):
            torch_dtype = dtype_hint
            if isinstance(data, np.ndarray):
                # If we do not know the NumPy partner for this torch dtype, leave as is
                if torch_dtype not in self._TORCH_TO_NP_DTYPE:
                    return data, torch_dtype

                target_np = np.dtype(self._TORCH_TO_NP_DTYPE[torch_dtype])

                # Special guard for uint64 -> int64. If overflow risk exists, fallback to float64.
                if target_np == np.dtype(np.int64) and data.dtype == np.dtype(np.uint64):
                    if data.size and data.max() > np.iinfo(np.int64).max:
                        return _cast_ndarray_minimal(data, np.dtype(np.float64)), self._module.float64
                    return _cast_ndarray_minimal(data, np.dtype(np.int64)), torch_dtype

                return _cast_ndarray_minimal(data, target_np), torch_dtype

            # No array given, just return dtype
            return data, torch_dtype

        # Case 2: resolve NumPy dtype from dtype_hint or from data
        npdt = None
        if dtype_hint is not None:
            try:
                npdt = np.dtype(dtype_hint)
            except TypeError:
                raise ValueError(f"Cannot convert {dtype_hint} to a NumPy dtype.")
        elif isinstance(data, np.ndarray):
            npdt = data.dtype

        # If we cannot resolve anything and no array, let torch infer
        if npdt is None and not isinstance(data, np.ndarray):
            return data, None

        # Convert unsupported NumPy dtype to a compatible one
        npdt_conv = self._convert_numpy_dtype_for_torch(npdt)

        # Map to torch dtype
        torch_dtype = self._NP_TO_TORCH_DTYPE.get(npdt_conv)
        if torch_dtype is None:
            # Known special cases
            if npdt == np.dtype(np.uint64):
                torch_dtype = self._module.int64
            elif str(npdt) in ("float128", "longdouble"):
                torch_dtype = self._module.float64
            else:
                raise ValueError(f"No corresponding torch dtype for NumPy dtype {npdt}.")

        # Case 3: data is an array, cast only if needed
        if isinstance(data, np.ndarray):
            # If current array dtype differs from the converted NumPy dtype, cast
            if data.dtype != npdt_conv:
                # Special guard for uint64 -> int64
                if data.dtype == np.dtype(np.uint64) and npdt_conv == np.dtype(np.int64):
                    if data.size and data.max() > np.iinfo(np.int64).max:
                        return _cast_ndarray_minimal(data, np.dtype(np.float64)), self._module.float64
                    
                    return _cast_ndarray_minimal(data, np.dtype(np.int64)), torch_dtype
                
                return _cast_ndarray_minimal(data, npdt_conv), torch_dtype

            # If dtypes match but mapping wants int64 for uint64, guard overflow
            if data.dtype == np.dtype(np.uint64) and torch_dtype == self._module.int64:
                if data.size and data.max() > np.iinfo(np.int64).max:
                    return _cast_ndarray_minimal(data, np.dtype(np.float64)), self._module.float64
                
                return _cast_ndarray_minimal(data, np.dtype(np.int64)), torch_dtype

            # No cast needed
            return data, torch_dtype

        # Not an ndarray. Return resolved dtype only.
        return data, torch_dtype

        
    def _ensure_torch_dtype(self, dtype_hint):
        """
        Normalize a user-provided dtype hint (NumPy dtype/class/torch.dtype/tuple)
        into a valid torch.dtype. Returns None if it cannot be resolved.
        """
        # Unwrap single-element tuples like (torch.float32,)
        if isinstance(dtype_hint, tuple) and len(dtype_hint) == 1:
            dtype_hint = dtype_hint[0]
    
        # Already a torch.dtype
        if isinstance(dtype_hint, self._module.dtype):
            return dtype_hint
    
        # Try to interpret as NumPy dtype (np.dtype, np scalar class, or string)
        try:
            npdt = np.dtype(dtype_hint)
        except Exception:
            return None
    
        # Convert unsupported NumPy dtype to a torch-compatible NumPy dtype
        npdt_conv = self._convert_numpy_dtype_for_torch(npdt)
    
        # Map to torch dtype
        torch_dtype = self._NP_TO_TORCH_DTYPE.get(npdt_conv)
        if torch_dtype is None:
            # Known special case: uint64 maps to int64 (with potential overflow concerns elsewhere)
            if npdt == np.dtype(np.uint64):
                return self._module.int64
            # Give up
            return None
    
        return torch_dtype
    
    def array(self, data, *args, **kwargs):
        """
        Wrapper around torch.tensor() with automatic dtype compatibility handling.
        If dtype is provided as a NumPy dtype, it will be mapped to torch.dtype.
        If data is a NumPy array with an unsupported dtype, it will be safely cast.
        """
        if 'device' not in kwargs:
            kwargs['device'] = self.device
    
        
        # Resolve dtype and cast data only when needed
        data, resolved_torch_dtype = self._np_dtype_to_torch_dtype(
            data=data,
            dtype_hint=kwargs.get("dtype", None)
        )

        # If we found a torch dtype, set it. Otherwise remove dtype and let torch infer.
        if resolved_torch_dtype is not None:
            kwargs["dtype"] = resolved_torch_dtype
        else:
            kwargs.pop("dtype", None)

        return self._module.tensor(data, *args, **kwargs)   
    
    def any(self, *args, **kwargs):
        """
        Wrapper for torch.any(). Note: torch.any does not accept dtype; remove it if present.
        """
        if 'device' not in kwargs:
            kwargs['device'] = self.device
    
        # torch.any does not take dtype; ensure we don't pass it
        kwargs.pop('dtype', None)
    
        return self._module.any(*args, **kwargs)
    
    
    def zeros(self, *args, **kwargs):
        """
        Wrapper for torch.zeros() that safely coerces dtype to a torch.dtype if provided.
        """
        if 'device' not in kwargs:
            kwargs['device'] = self.device
    
        if 'dtype' in kwargs:
            resolved = self._ensure_torch_dtype(kwargs['dtype'])
            if resolved is not None:
                kwargs['dtype'] = resolved
            else:
                # If we cannot resolve to a torch.dtype, let torch infer by dropping dtype
                kwargs.pop('dtype')
    
        return self._module.zeros(*args, **kwargs)
    
    
    def ones(self, *args, **kwargs):
        """
        Wrapper for torch.ones() that safely coerces dtype to a torch.dtype if provided.
        """
        if 'device' not in kwargs:
            kwargs['device'] = self.device
    
        if 'dtype' in kwargs:
            resolved = self._ensure_torch_dtype(kwargs['dtype'])
            if resolved is not None:
                kwargs['dtype'] = resolved
            else:
                kwargs.pop('dtype')
    
        return self._module.ones(*args, **kwargs)


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