import unittest

import numpy as np

from tools.temporal_filter import TemporalMaskFilter


class TemporalMaskFilterTests(unittest.TestCase):
    def make_filter(self, **overrides):
        cfg = dict(
            HISTORY_LENGTH=5,
            MIN_HIT_FRAMES=3,
            MIN_HISTORY_FRAMES=3,
            MOTION_TOLERANCE_DILATE_ITER=0,
            KEEP_CURRENT_ONLY=True,
            WARMUP_MODE="passthrough",
        )
        cfg.update(overrides)
        return TemporalMaskFilter(cfg)

    def test_repeated_pixel_survives_after_three_hits(self):
        temporal_filter = self.make_filter()
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        temporal_filter.update(mask)
        temporal_filter.update(mask)
        result = temporal_filter.update(mask)

        self.assertTrue(result["temporal_ready"])
        self.assertTrue(result["temporal_mask"][2, 2])

    def test_single_frame_noise_is_removed_after_warmup(self):
        temporal_filter = self.make_filter()
        empty = np.zeros((5, 5), dtype=bool)
        noise = empty.copy()
        noise[2, 2] = True

        temporal_filter.update(empty)
        temporal_filter.update(empty)
        result = temporal_filter.update(noise)

        self.assertFalse(result["temporal_mask"][2, 2])

    def test_motion_tolerance_accepts_small_position_change(self):
        temporal_filter = self.make_filter(MOTION_TOLERANCE_DILATE_ITER=1)
        masks = []
        for x in (2, 2, 3):
            mask = np.zeros((5, 5), dtype=bool)
            mask[2, x] = True
            masks.append(mask)

        temporal_filter.update(masks[0])
        temporal_filter.update(masks[1])
        result = temporal_filter.update(masks[2])

        self.assertTrue(result["temporal_mask"][2, 3])


if __name__ == "__main__":
    unittest.main()
