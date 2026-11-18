"""Focus pipeline: prompt mapping, shape guards and look-alike rejection."""
import unittest

import numpy as np

from scanners.focus import focus_detections


class FocusPipelineTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def run_focus(self, world, coco=None):
        return focus_detections(
            self.image,
            0.35,
            False,
            coco_raw=coco or [],
            world_raw=world,
            coco_all=coco or [],
        )

    def test_charger_adapter_prompt_becomes_a_detection(self):
        detections = self.run_focus([("phone charger", 0.22, [100, 100, 260, 260])])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "Charger Adapter")
        self.assertEqual(detections[0]["reference_key"], "charger-adapter")

    def test_charger_cable_prompt_becomes_a_detection(self):
        detections = self.run_focus([("charger cable", 0.18, [100, 200, 500, 240])])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "Charger Cable")

    def test_phone_prompt_rejects_charger_lookalike(self):
        detections = self.run_focus([
            ("smartphone", 0.30, [100, 100, 260, 260]),
            ("phone charger", 0.22, [100, 100, 260, 260]),
        ])
        self.assertEqual(detections, [])

    def test_coco_mouse_survives_at_low_confidence(self):
        coco = [{
            "raw": "mouse",
            "class_id": 64,
            "confidence": 0.16,
            "bbox": [0.10, 0.10, 0.20, 0.15],
        }]
        detections = self.run_focus([], coco=coco)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "Computer Mouse")

    def test_frame_sized_box_is_rejected(self):
        detections = self.run_focus([("charger adaptor", 0.40, [0, 0, 640, 480])])
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
