import numpy as np


try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as err:
    print("[WARNING] Cannot use GPU computing based on CuPy.")

try:
    import jax
    from jax import device_put
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError) as err:
    print("[WARNING] Cannot use GPU computing based on Jax")


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
            if str.isnumeric(options[-1]):
                _device, _device_id = options

                if _device in ["cuda", "cupy"]:
                    _backend = "cupy"
                    _device = "gpu"


            elif isinstance(options[-1], str):
                _backend, _device = options
                _device_id = 0

        elif len(options) == 3:
            _backend, _device, _device_id = options

            if _backend not in ["numpy", "cupy", "jax"]:
                raise ValueError("backend should be one of 'numpy', 'cupy', or "\
                                 "'jax', not %s" % (_backend))
        else:
            raise RuntimeError("Illegal device:", device)

        _device_id = int(_device_id)

    else:
        if device == "cupy":
            _backend = "cupy"
            _device = "gpu"
            _device_id = 0
        elif device == "jax":
            _backend = "jax"
            _device = "gpu"
            _device_id = 0
        elif device == "numpy":
            _backend = "numpy"
            _device = "cpu"
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
    elif _backend == "jax" and _device in ["gpu", "cuda"]:
        return JaxModule(_device, _device_id)
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

        self._device = cp.cuda.Device()
        self._device.id = self._device_id
        self._device.use()

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def any(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.ones(*args, **kwargs)

    def array(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.array(*args, **kwargs)

    def abs(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.abs(*args, **kwargs)

    def get(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()

        return arr

    def is_array(self, obj):
        return isinstance(obj, (cp.ndarray, cp.generic))

    def isnan(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.isinf(*args, **kwargs)


class JaxModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)
        self._device = jax.devices()[device_id]

    def any(self, *args, **kwargs):
        return jnp.any(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return device_put(jnp.zeros(*args, **kwargs), self._device)

    def ones(self, *args, **kwargs):
        return device_put(jnp.ones(*args, **kwargs), self._device)

    def array(self, *args, **kwargs):
        return device_put(jnp.array(*args, **kwargs), self._device)

    def abs(self, *args, **kwargs):
        return jnp.abs(*args, **kwargs)

    def get(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()

        return arr

    def set(self, arr, ind, val):
        return arr.at[ind].set(val)

    def is_array(self, obj):
        return isinstance(obj, (jnp.ndarray, jnp.generic))

    def isnan(self, *args, **kwargs):
        return jnp.isnan(*args, **kwargs)

    def isinf(self, *args, **kwargs):
        return jnp.isinf(*args, **kwargs)
