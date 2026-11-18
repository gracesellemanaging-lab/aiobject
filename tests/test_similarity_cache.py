"""Similarity cache housekeeping should never delete in-flight downloads."""
import os
import tempfile
import unittest
from pathlib import Path

from ai import similarity


class PruneCacheTest(unittest.TestCase):
    def test_keeps_newest_files_and_skips_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = similarity.SIMILAR_DIR
            similarity.SIMILAR_DIR = Path(tmp)
            try:
                partial = Path(tmp) / "active.part.jpg"
                partial.write_bytes(b"partial")
                for index in range(5):
                    path = Path(tmp) / f"{index}.jpg"
                    path.write_bytes(b"image")
                    os.utime(path, (1000 + index, 1000 + index))
                similarity.VisualSimilarityService()._prune_cache(keep=2)
                remaining = {path.name for path in Path(tmp).glob("*.jpg")}
            finally:
                similarity.SIMILAR_DIR = original
        self.assertEqual(remaining, {"3.jpg", "4.jpg", "active.part.jpg"})


if __name__ == "__main__":
    unittest.main()
