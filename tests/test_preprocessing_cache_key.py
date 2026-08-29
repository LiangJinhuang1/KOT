import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import source_stamp


class SourceStampTests(unittest.TestCase):
    """The cache key must follow the bytes at a path, not just the path.

    The velocity pipeline rewrites the same filename, so "same path, new contents"
    is the normal case; a path-only key served the stale cache with no warning.
    """

    def test_stamp_includes_size_and_mtime(self):
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
            handle.write(b"velocity")
            path = handle.name
        try:
            stamp = source_stamp(path)
            self.assertIn(os.path.abspath(path), stamp)
            self.assertIn("size=8", stamp)
            self.assertIn("mtime=", stamp)
        finally:
            os.unlink(path)

    def test_regenerating_in_place_changes_the_stamp(self):
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
            handle.write(b"first velocity run")
            path = handle.name
        try:
            before = source_stamp(path)
            time.sleep(1.1)   # mtime resolution
            with open(path, "wb") as handle:
                handle.write(b"regenerated velocity run")
            after = source_stamp(path)
            self.assertNotEqual(before, after)
        finally:
            os.unlink(path)

    def test_same_file_untouched_keeps_its_stamp(self):
        """Cache hits must still happen, or every run reprocesses 9 GB."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
            handle.write(b"velocity")
            path = handle.name
        try:
            self.assertEqual(source_stamp(path), source_stamp(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
