"""Guards that protect the API from hostile or broken uploads."""
import io
import unittest

from PIL import Image

from ai.utils import decode_image


class DecodeImageGuardTest(unittest.TestCase):
    def test_normal_photo_decodes(self):
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), "white").save(buffer, format="JPEG")
        self.assertEqual(decode_image(buffer.getvalue()).shape[:2], (480, 640))

    def test_oversized_pixels_are_rejected(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8000, 8000), "white").save(buffer, format="PNG")
        with self.assertRaisesRegex(ValueError, "too large"):
            decode_image(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
