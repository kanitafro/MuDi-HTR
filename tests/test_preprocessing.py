"""Focused tests for preprocessing placeholders."""

import tempfile
import unittest
import importlib.util
from pathlib import Path

from preprocessing.online_preprocess import parse_didi

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None


class TestOnlinePreprocess(unittest.TestCase):
    def test_parse_didi_json_normalizes_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.json"
            path.write_text(
                '{"strokes": [[{"x": 10, "y": 20}, {"x": 30, "y": 40}]]}',
                encoding="utf-8",
            )
            strokes = parse_didi(path)

        self.assertEqual(len(strokes), 1)
        self.assertEqual(strokes[0][0], (0.0, 0.0))
        self.assertEqual(strokes[0][1], (1.0, 1.0))


class TestOfflinePreprocess(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY and HAS_PIL, "numpy and pillow are required")
    def test_preprocess_image_shape_and_range(self) -> None:
        import numpy as np
        from PIL import Image

        from preprocessing.offline_preprocess import preprocess_image

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.png"
            Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(path)
            processed = preprocess_image(path, image_size=(32, 64))

        self.assertEqual(processed.shape, (32, 64))
        self.assertTrue(float(processed.min()) >= 0.0)
        self.assertTrue(float(processed.max()) <= 1.0)


if __name__ == "__main__":
    unittest.main()
