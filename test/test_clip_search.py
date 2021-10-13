import logging
import unittest

import numpy as np
from PIL import Image
from clip_search.search import ClipSearch

format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
# set up logging to console
console = logging.StreamHandler()
console.setLevel("DEBUG")
formatter = logging.Formatter(format)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

RGBS = [
    np.zeros((1080, 1920, 3), np.uint8),
    255 * np.ones((1080, 1920, 3), np.uint8),
]
RGBS_PILS = [Image.fromarray(x) for x in RGBS]
TEXTS = ["a black image", "a white image"]


class TestModule(unittest.TestCase):
    def test_image_encode(self):
        clipsearch = ClipSearch(model_name="RN50")
        feats = clipsearch.encode_images(RGBS_PILS)
        logger.info(f"image features shape {feats.shape}")
        return True

    def test_image_encode_np(self):
        clipsearch = ClipSearch(model_name="RN50")
        feats = clipsearch.encode_images(RGBS)
        logger.info(f"image features shape {feats.shape}")
        return True

    def test_text_encode(self):
        clipsearch = ClipSearch(model_name="RN50")
        feats = clipsearch.encode_texts(TEXTS)
        logger.info(f"text features shape {feats.shape}")
        return True

    def test_query_with_feats(self):
        clipsearch = ClipSearch(model_name="RN50")
        img_feats = clipsearch.encode_images(RGBS)
        text_feats = clipsearch.encode_texts(TEXTS)
        logger.info(clipsearch.query_with_feats(img_feats[0], text_feats))
        logger.info(clipsearch.query_with_feats(img_feats, text_feats[0]))
        return True


if __name__ == "__main__":
    unittest.main()
