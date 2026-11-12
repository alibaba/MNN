import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.vision import Lfm3VlVision, Qwen2Vision


class Tokenizer:
    def __init__(self, family):
        self.tokens = {f'<|{family}_start|>': [125009], f'<|{family}_end|>': [125010]}

    def encode(self, text, add_special_tokens=False):
        return self.tokens.get(text, [36, 100, 99])


class MediaMarkerTest(unittest.TestCase):
    def test_small_image_resize_is_family_specific(self):
        for height, width, expected in [(16, 16, (256, 256)), (16, 64, (128, 512)),
                                         (64, 16, (512, 128)), (31, 31, (256, 256)),
                                         (32, 32, (256, 256)), (33, 33, (256, 256))]:
            with self.subTest(height=height, width=width):
                self.assertEqual(Lfm3VlVision.smart_resize(None, height, width), expected)
        for height, width in [(0, 16), (-1, 16), (1, 201)]:
            with self.assertRaises(ValueError):
                Lfm3VlVision.smart_resize(None, height, width)
        with self.assertRaises(ValueError):
            Qwen2Vision.smart_resize(None, 16, 16, 32, 65536, 16777216)

    def test_processor_markers_and_tokenizer_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'processor_config.json'
            config.write_text(json.dumps({'vision_start_token': '<|media_start|>',
                                          'vision_end_token': '<|media_end|>'}))
            for family in ['vision', 'media']:
                with self.subTest(family=family):
                    result = Lfm3VlVision.media_markers(directory, Tokenizer(family))
                    self.assertEqual(result['start'], (f'<|{family}_start|>', 125009))
                    self.assertEqual(result['end'], (f'<|{family}_end|>', 125010))
            with self.assertRaises(ValueError):
                Lfm3VlVision.media_markers(directory, Tokenizer('missing'))


if __name__ == '__main__':
    unittest.main()
