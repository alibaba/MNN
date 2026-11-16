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
    def test_video_rejected_before_tokenization(self):
        visual = object.__new__(Lfm3VlVision)
        with self.assertRaises(NotImplementedError):
            visual.str_to_ids('<video>missing.mp4</video>Describe')

    def test_size_override_and_native_size(self):
        from unittest.mock import patch
        import torch
        from PIL import Image
        visual = object.__new__(Lfm3VlVision)
        torch.nn.Module.__init__(visual)
        visual._image_size_overrides = [(64, 96), None]
        with patch.object(Qwen2Vision, 'image_to_tensor', lambda self, image: (self.image_height, self.image_width)):
            self.assertEqual(visual.image_to_tensor(Image.new('RGB', (160, 80))), (64, 96))
            self.assertEqual(visual.image_to_tensor(Image.new('RGB', (160, 80))), (80, 160))

    def test_compact_single_image_and_isolated_multiple_grids(self):
        import torch
        visual = object.__new__(Lfm3VlVision)
        mask = visual.vision_attention_mask(torch.tensor([[1, 2, 2]]))
        self.assertEqual(tuple(mask.shape), (1,))
        self.assertEqual(mask.item(), 0)
        mask = visual.vision_attention_mask(torch.tensor([[1, 2, 2], [1, 2, 2]]))
        self.assertEqual(tuple(mask.shape), (1, 8, 8))
        self.assertEqual(mask[0, 0, 1].item(), 0)
        self.assertLess(mask[0, 0, 4].item(), -1e30)

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
