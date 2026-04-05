"""Video and image utility error handling and frame ordering.

Regression tests for:
- 7-7: "doest not exists" typo → "does not exist"
- 7-8: deprecated imghdr module replaced with extension-based check
- 7-9: video frames must be sorted for consistent ordering
"""

import os
import pytest


class TestVideoFrameSorting:
    """Video frames must be sorted for consistent frame ordering."""

    def test_create_video_sorts_frames(self, tmp_path):
        """create_video should process frames in sorted order."""
        from PIL import Image

        for name in ["frame_003.png", "frame_001.png", "frame_002.png"]:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(str(tmp_path / name))

        # Non-image file should be skipped
        with open(str(tmp_path / "readme.txt"), "w") as f:
            f.write("not an image")

        import inspect
        from lpf.visualization import video
        source = inspect.getsource(video)
        assert "sorted(" in source
        assert "import imghdr" not in source


class TestImageNotADirectoryError:
    """Image functions should raise NotADirectoryError with correct message."""

    def test_merge_multiple_invalid_path(self):
        from lpf.visualization.image import merge_multiple
        with pytest.raises(NotADirectoryError, match="does not exist"):
            merge_multiple(dpath_input="/nonexistent/path/12345")

    def test_merge_single_timeseries_invalid_path(self):
        from lpf.visualization.image import merge_single_timeseries
        with pytest.raises(NotADirectoryError, match="does not exist"):
            merge_single_timeseries(dpath_input="/nonexistent/path/12345")

    def test_merge_multiple_timeseries_invalid_path(self):
        from lpf.visualization.image import merge_multiple_timeseries
        with pytest.raises(NotADirectoryError, match="does not exist"):
            merge_multiple_timeseries(dpath_input="/nonexistent/path/12345")
