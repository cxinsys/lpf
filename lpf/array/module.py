import numpy as np


try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as err:
    print("[WARNING] Cannot use GPU computing based on CuPy.")


def parse_device(device):
    if device is None:
        return "cpu", None

    device = device.lower()
    _device = device
    _device_id = 0

    if ":" in device:
        _device, _device_id = device.split(":")
        _device_id = int(_device_id)

    if _device not in ["cpu", "gpu", "cuda"]:
        raise ValueError("device should be one of 'cpu', " \
                         "'gpu', or 'cuda', not %s" % (device))

    return _device, _device_id


def get_array_module(device):
    _device, _device_id = parse_device(device)

    if "gpu" in _device or "cuda" in _device:
        return CupyModule(_device, _device_id)
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

    def isnan(self, *args, **kwargs):
        return np.isnan(*args, **kwargs)

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

    def is_array(self, obj):
        return isinstance(obj, (np.ndarray, np.generic))


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

    def isnan(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.isnan(*args, **kwargs)

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
        return arr.get()

    def is_array(self, obj):
        return isinstance(obj, (cp.ndarray, cp.generic))