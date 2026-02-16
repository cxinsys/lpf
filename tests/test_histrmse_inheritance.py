"""MaxHistogramRootMeanSquareError must inherit from
EachHistogramRootMeanSquareError, not Objective.

Before fix: super().compute() called Objective.compute() which raises
NotImplementedError — any use of MaxHistogramRootMeanSquareError crashed.
"""

import numpy as np
import pytest

from lpf.objectives.histrmse import (
    EachHistogramRootMeanSquareError,
    MaxHistogramRootMeanSquareError,
    SumHistogramRootMeanSquareError,
    MeanHistogramRootMeanSquareError,
    MinHistogramRootMeanSquareError,
)
from lpf.objectives import Objective


class TestMaxHistogramRootMeanSquareErrorInheritance:

    def test_inherits_from_each(self):
        """MaxHistogramRootMeanSquareError should be a subclass of
        EachHistogramRootMeanSquareError, not Objective directly."""
        assert issubclass(MaxHistogramRootMeanSquareError,
                          EachHistogramRootMeanSquareError)

    def test_all_siblings_share_same_parent(self):
        """All aggregate variants should inherit from EachHistogramRootMeanSquareError."""
        for cls in [SumHistogramRootMeanSquareError,
                    MeanHistogramRootMeanSquareError,
                    MinHistogramRootMeanSquareError,
                    MaxHistogramRootMeanSquareError]:
            assert issubclass(cls, EachHistogramRootMeanSquareError), \
                f"{cls.__name__} does not inherit from EachHistogramRootMeanSquareError"

    def test_compute_does_not_raise(self):
        """MaxHistogramRootMeanSquareError.compute() should not raise
        NotImplementedError."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)
        target = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)

        obj = MaxHistogramRootMeanSquareError(targets=[target])
        result = obj.compute(img)

        assert isinstance(result, (float, np.floating)), \
            f"Expected scalar, got {type(result)}"
        assert result >= 0, "RMSE should be non-negative"

    def test_max_ge_min(self):
        """Max aggregate should be >= Min aggregate for the same inputs."""
        rng = np.random.RandomState(123)
        img = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)
        targets = [
            rng.randint(0, 256, (64, 64, 3)).astype(np.uint8),
            rng.randint(0, 256, (64, 64, 3)).astype(np.uint8),
        ]
        max_obj = MaxHistogramRootMeanSquareError(targets=targets)
        min_obj = MinHistogramRootMeanSquareError(targets=targets)

        max_val = max_obj.compute(img)
        min_val = min_obj.compute(img)

        assert max_val >= min_val
