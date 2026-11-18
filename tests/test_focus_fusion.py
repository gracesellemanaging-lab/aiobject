"""Fusion rules for the focus objects (no model needed)."""
import unittest

from ai.focus_fusion import area_ok, fuse, nms
from ai.focus_targets import COCO_TO_KEY, FOCUS_PROMPTS, FOCUS_TARGETS, PROMPT_TO_KEY

BOX = [0.30, 0.30, 0.20, 0.20]
NEAR = [0.31, 0.31, 0.20, 0.20]
FAR = [0.70, 0.70, 0.10, 0.10]


def c(key, conf, bbox=BOX):
    return {"key": key, "conf": conf, "bbox": bbox}


class FocusFusionTest(unittest.TestCase):
    def test_targets_cover_requested_objects(self):
        self.assertEqual(set(FOCUS_TARGETS), {"mouse", "remote", "keyboard", "ballpen", "adapter", "connector"})
        self.assertEqual(COCO_TO_KEY, {"mouse": "mouse", "remote": "remote", "keyboard": "keyboard"})
        self.assertEqual(len(FOCUS_PROMPTS), len(set(FOCUS_PROMPTS)))
        self.assertIn("ballpoint pen", PROMPT_TO_KEY)

    def test_weak_coco_hit_is_kept_when_world_agrees(self):
        # 0.17 is below mouse min_conf (0.25) on its own, but both models see it.
        out = fuse([c("mouse", 0.17)], [c("mouse", 0.16, NEAR)], [], 0.35)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["sources"], ["coco", "world"])
        self.assertGreater(out[0]["conf"], 0.30)

    def test_weak_single_model_hit_is_rejected(self):
        self.assertEqual(fuse([c("mouse", 0.17)], [], [], 0.35), [])

    def test_slider_can_only_lower_threshold(self):
        # slider at 0.90 must not hide a 0.30 mouse (its own floor is 0.25)
        self.assertEqual(len(fuse([c("mouse", 0.30)], [], [], 0.90)), 1)
        # slider at 0.15 lets a 0.16 ballpen through
        self.assertEqual(len(fuse([], [c("ballpen", 0.16)], [], 0.15)), 1)

    def test_lookalike_beats_remote(self):
        out = fuse([c("remote", 0.30)], [], [c("remote", 0.45)], 0.35)
        self.assertEqual(out, [])

    def test_lookalike_far_away_does_not_hide_remote(self):
        out = fuse([c("remote", 0.30)], [], [c("remote", 0.45, FAR)], 0.35)
        self.assertEqual(len(out), 1)

    def test_one_label_per_object(self):
        out = fuse([], [c("adapter", 0.40), c("connector", 0.22, NEAR)], [], 0.35)
        self.assertEqual([d["key"] for d in out], ["adapter"])

    def test_duplicate_prompts_collapse(self):
        self.assertEqual(len(nms([c("ballpen", 0.30), c("ballpen", 0.25, NEAR)])), 1)

    def test_area_limits(self):
        self.assertFalse(area_ok("ballpen", [0, 0, 0.9, 0.9]))      # whole frame is not a pen
        self.assertFalse(area_ok("ballpen", [0, 0, 0.01, 0.01]))    # a speck is not a pen
        self.assertTrue(area_ok("ballpen", [0, 0, 0.20, 0.03]))
        self.assertTrue(area_ok("keyboard", [0.1, 0.5, 0.8, 0.4]))


if __name__ == "__main__":
    unittest.main()
