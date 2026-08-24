import unittest

import torch
from PIL import Image

from starVLA.model.tools import preprocess_images


class PreprocessImagesTest(unittest.TestCase):
    def test_different_view_resolutions_are_padded_before_stacking(self):
        images = [
            [
                Image.new("RGB", (640, 480), "red"),
                Image.new("RGB", (640, 320), "blue"),
            ]
        ]

        result = preprocess_images(images, target_size=224, mode="crop")

        self.assertEqual(result.shape, (1, 2, 3, 168, 224))
        self.assertTrue(torch.all(result[0, 1, :, :28, :] == 1))
        self.assertTrue(torch.all(result[0, 1, :, -28:, :] == 1))

    def test_different_sample_resolutions_are_padded_before_stacking(self):
        images = [
            [Image.new("RGB", (640, 480), "red")],
            [Image.new("RGB", (640, 320), "blue")],
        ]

        result = preprocess_images(images, target_size=224, mode="crop")

        self.assertEqual(result.shape, (2, 1, 3, 168, 224))


if __name__ == "__main__":
    unittest.main()
