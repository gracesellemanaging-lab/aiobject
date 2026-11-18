"""Priority-object fusion rules: thresholds, shape guards and look-alikes."""
import unittest

from ai.focus_fusion import fuse, shape_ok, threshold
from ai.focus_targets import (
    COCO_TO_KEY,
    FOCUS_PROMPTS,
    FOCUS_TARGETS,
    NEGATIVE_PROMPTS,
    PROMPT_TO_KEY,
)


class FocusTargetsTest(unittest.TestCase):
    def test_requested_objects_have_prompts(self):
        labels = {target.label for target in FOCUS_TARGETS.values()}
        self.assertIn("Computer Mouse", labels)
        self.assertIn("Remote Control", labels)
        self.assertIn("Keyboard", labels)
        self.assertIn("Charger Adapter", labels)
        self.assertIn("Charger Cable", labels)
        self.assertIn("Charging Connector", labels)
        self.assertIn("charger cable", PROMPT_TO_KEY)
        self.assertIn("charger adaptor", PROMPT_TO_KEY)
        self.assertIn("type-c cable", PROMPT_TO_KEY)
        self.assertIn("remote", COCO_TO_KEY)

    def test_every_prompt_maps_to_a_known_key(self):
        for prompt, key in PROMPT_TO_KEY.items():
            self.assertIn(key, FOCUS_TARGETS, prompt)

    def test_negative_prompts_are_part_of_the_world_vocabulary(self):
        for prompt in NEGATIVE_PROMPTS:
            self.assertIn(prompt, FOCUS_PROMPTS)

    def test_slider_never_makes_priority_objects_stricter(self):
        for key, target in FOCUS_TARGETS.items():
            self.assertAlmostEqual(threshold(key, 0.9), target.min_conf)
            self.assertAlmostEqual(threshold(key, 0.01), 0.01)


class ShapeGuardTest(unittest.TestCase):
    def test_tiny_specks_are_rejected(self):
        self.assertFalse(shape_ok("mouse", [0.5, 0.5, 0.002, 0.002], 640, 480))

    def test_frame_sized_boxes_are_rejected(self):
        self.assertFalse(shape_ok("mouse", [0.0, 0.0, 1.0, 1.0], 640, 480))

    def test_long_thin_cable_is_accepted(self):
        self.assertTrue(shape_ok("charger_cable", [0.1, 0.4, 0.6, 0.05], 640, 480))

    def test_cable_box_with_impossible_aspect_is_rejected(self):
        self.assertFalse(shape_ok("charger_cable", [0.1, 0.4, 0.7, 0.002], 640, 480))


class FusionTest(unittest.TestCase):
    def test_coco_and_world_agreement_is_merged_and_boosted(self):
        coco = [{"key": "mouse", "conf": 0.16, "bbox": [0.10, 0.10, 0.20, 0.15]}]
        world = [{"key": "mouse", "conf": 0.14, "bbox": [0.10, 0.10, 0.20, 0.15]}]
        result = fuse(coco, world, [], confidence=0.35)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sources"], ["coco", "world"])
        self.assertGreater(result[0]["conf"], 0.20)

    def test_weak_single_model_detection_is_rejected(self):
        world = [{"key": "charger_cable", "conf": 0.06, "bbox": [0.1, 0.4, 0.6, 0.05]}]
        self.assertEqual(fuse([], world, [], confidence=0.35), [])

    def test_coco_only_hit_survives_slightly_below_floor(self):
        coco = [{"key": "mouse", "conf": 0.16, "bbox": [0.10, 0.10, 0.20, 0.15]}]
        result = fuse(coco, [], [], confidence=0.35)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sources"], ["coco"])

    def test_phone_lookalike_suppresses_remote(self):
        coco = [{"key": "remote", "conf": 0.25, "bbox": [0.1, 0.1, 0.3, 0.3]}]
        negatives = [{"key": "remote", "conf": 0.62, "bbox": [0.1, 0.1, 0.3, 0.3]}]
        self.assertEqual(fuse(coco, [], negatives, confidence=0.35), [])

    def test_same_object_is_reported_once(self):
        world = [
            {"key": "charger_adapter", "conf": 0.21, "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"key": "charger_adapter", "conf": 0.19, "bbox": [0.11, 0.1, 0.2, 0.2]},
        ]
        result = fuse([], world, [], confidence=0.35)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["conf"], 0.21)


if __name__ == "__main__":
    unittest.main()
