import unittest

import numpy as np

from tools.appearance_change import postprocess_appearance_mask


class AppearancePostprocessTests(unittest.TestCase):
    def test_area_filter_removes_small_and_large_regions(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[1:3, 1:3] = True
        mask[10:15, 10:15] = True
        mask[20:30, 20:30] = True
        cfg = dict(
            ENABLE_APPEARANCE_MORPH=False,
            APPEARANCE_KERNEL_SIZE=(3, 3),
            APPEARANCE_OPEN_ITER=0,
            APPEARANCE_CLOSE_ITER=0,
            ENABLE_APPEARANCE_AREA_FILTER=True,
            APPEARANCE_MIN_REGION_AREA=10,
            APPEARANCE_MAX_REGION_AREA=50,
        )

        result = postprocess_appearance_mask(mask, cfg)

        self.assertFalse(result[1, 1])
        self.assertTrue(result[12, 12])
        self.assertFalse(result[25, 25])


if __name__ == "__main__":
    unittest.main()
