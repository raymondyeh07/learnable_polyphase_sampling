# Smoke tests for DDACSegmentation wrapper
import unittest

from models.base_segmentation import DDACSegmentation, DDAC_MODEL_MAP


class TestDDACSegmentation(unittest.TestCase):

    def test_ddac_model_creation(self):
        for model_name in DDAC_MODEL_MAP.keys():
            with self.subTest(model_name=model_name):
                model = DDACSegmentation(model_name=model_name)
                self.assertIsNotNone(model)
