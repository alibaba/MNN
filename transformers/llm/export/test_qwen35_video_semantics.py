import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image

sys.modules.setdefault("yaspin", types.SimpleNamespace(yaspin=lambda *args, **kwargs: None))

from utils.vision import Qwen2Vision


class DummyTokenizer:
    def __init__(self):
        self.last_text = ""
        self.special = {
            "<|vision_start|>": 248053,
            "<|vision_end|>": 248054,
            "<|image_pad|>": 248056,
            "<|video_pad|>": 248057,
        }

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        ids = []
        i = 0
        specials = sorted(self.special.items(), key=lambda item: len(item[0]), reverse=True)
        while i < len(text):
            matched = False
            for token, token_id in specials:
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(1000 + ord(text[i]) % 100)
                i += 1
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def make_qwen_vision():
    vision = object.__new__(Qwen2Vision)
    vision.tokenizer = DummyTokenizer()
    vision.temporal_patch_size = 2
    vision.patch_size = 14
    vision.merge_size = 2
    vision.vision_start_id = 248053
    vision.vision_end_id = 248054
    vision.image_pad_id = 248056
    vision.video_pad_id = 248057
    vision.vision_start_token = "<|vision_start|>"
    vision.vision_end_token = "<|vision_end|>"
    vision.image_pad_token = "<|image_pad|>"
    vision.video_pad_token = "<|video_pad|>"
    vision.image_embeds = []
    vision.video_embeds = []
    vision.image_grid_thw = []
    vision.vision_segments = []
    vision.video_fps = 2.0
    vision.video_max_frames = 768
    vision.video_max_pixels = 768 * 28 * 28
    vision.video_max_vision_tokens = 4096
    vision.image_height = 448
    vision.image_width = 448
    vision.min_pixels = 56 * 56
    vision.max_pixels = 14 * 14 * 4 * 1280
    return vision


def test_video_prompt_expands_to_timestamped_video_pads():
    vision = make_qwen_vision()

    def fake_video_process(path):
        assert path == "demo.mp4"
        vision.vision_segments.extend([
            {"type": "video", "grid": [1, 4, 4]},
            {"type": "video", "grid": [1, 4, 4]},
        ])
        return [(0.5, 4), (1.0, 4)]

    vision.video_process = fake_video_process

    input_ids = Qwen2Vision.str_to_ids(vision, "describe <video>demo.mp4</video> now")

    assert "<0.5 seconds><|vision_start|>" in vision.tokenizer.last_text
    assert "<1.0 seconds><|vision_start|>" in vision.tokenizer.last_text
    assert (input_ids == vision.video_pad_id).sum().item() == 8
    assert (input_ids == vision.image_pad_id).sum().item() == 0


def test_video_position_ids_use_video_grid_not_text_positions():
    vision = make_qwen_vision()
    vision.vision_segments.append({"type": "video", "grid": [1, 4, 4]})
    input_ids = torch.tensor([[
        11,
        vision.vision_start_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.vision_end_id,
        12,
    ]])

    position_ids = Qwen2Vision.get_position_ids(vision, input_ids, input_ids.numel(), 0)

    expected = torch.tensor([
        [0, 1, 2, 2, 2, 2, 4, 5],
        [0, 1, 2, 2, 3, 3, 4, 5],
        [0, 1, 2, 3, 2, 3, 4, 5],
    ], dtype=torch.int)
    assert torch.equal(position_ids, expected)


def test_image_prompt_still_uses_image_pads():
    vision = make_qwen_vision()

    def fake_img_process(image):
        vision.image_grid_thw.append([1, 4, 4])
        return 4

    vision.img_process = fake_img_process

    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        Image.new("RGB", (32, 32)).save(image_file.name)
        input_ids = Qwen2Vision.str_to_ids(vision, f"describe <img>{image_file.name}</img> now")

    assert "<|vision_start|><|image_pad|>" in vision.tokenizer.last_text
    assert (input_ids == vision.image_pad_id).sum().item() == 4
    assert (input_ids == vision.video_pad_id).sum().item() == 0


def test_video_position_ids_advance_by_largest_axis_for_non_square_grid():
    vision = make_qwen_vision()
    vision.vision_segments.append({"type": "video", "grid": [1, 8, 4]})
    input_ids = torch.tensor([[
        11,
        vision.vision_start_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.video_pad_id,
        vision.vision_end_id,
        12,
    ]])

    position_ids = Qwen2Vision.get_position_ids(vision, input_ids, input_ids.numel(), 0)

    assert position_ids[0, -2].item() == 6
    assert position_ids[0, -1].item() == 7


def test_video_sampling_matches_qwen3vl_min_frames_and_linspace():
    vision = make_qwen_vision()

    assert Qwen2Vision.video_sample_indices(vision, 25, 25.0) == [0, 8, 16, 24]
    assert Qwen2Vision.video_sample_indices(vision, 3, 25.0) == [0, 1, 1, 2]
    vision.video_max_frames = 5
    assert len(Qwen2Vision.video_sample_indices(vision, 100, 10.0)) == 4
    assert Qwen2Vision.video_sample_indices(vision, 1, 25.0) == [0, 0]
    vision.video_max_frames = 1
    assert Qwen2Vision.video_sample_indices(vision, 1, 25.0) == []


def test_video_resize_preserves_aspect_and_caps_pixels():
    vision = make_qwen_vision()

    width, height = Qwen2Vision.video_resize_size(vision, 3840, 2160)

    assert width % (vision.patch_size * vision.merge_size) == 0
    assert height % (vision.patch_size * vision.merge_size) == 0
    assert width * height <= vision.video_max_pixels
    assert abs((width / height) - (3840 / 2160)) < 0.1
    vision.video_max_pixels = 0
    assert Qwen2Vision.video_resize_size(vision, 854, 854) == (840, 840)


def test_video_resize_uses_total_token_budget_for_long_video():
    vision = make_qwen_vision()
    frame_count = len(Qwen2Vision.video_sample_indices(vision, 250, 25.0))

    width, height = Qwen2Vision.video_resize_size(vision, 3840, 2160, frame_count)
    seq_len = (
        frame_count // vision.temporal_patch_size * (height // vision.patch_size) * (width // vision.patch_size)
    )

    assert frame_count == 20
    assert seq_len <= vision.video_max_vision_tokens
    assert width * height <= vision.video_max_pixels
    assert Qwen2Vision.video_effective_max_pixels(vision, frame_count) <= vision.video_max_pixels


def test_video_attention_mask_rejects_over_budget_before_allocation():
    vision = make_qwen_vision()
    vision.video_max_vision_tokens = 4

    try:
        Qwen2Vision.vision_attention_mask(
            vision, torch.tensor([[2, 4, 4]]), max_seq_len=vision.video_max_vision_tokens
        )
    except RuntimeError as exc:
        assert "video_max_vision_tokens" in str(exc)
    else:
        raise AssertionError("over-budget video mask should fail before allocation")


def test_video_process_uses_first_frame_size_when_metadata_is_missing():
    vision = make_qwen_vision()
    old_cv2 = sys.modules.get("cv2")
    seen_sizes = []

    class FakeCapture:
        def __init__(self, path):
            self.read_count = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 0:
                return 25.0
            if prop == 1:
                return 1
            return 0

        def read(self):
            if self.read_count > 0:
                return False, None
            self.read_count += 1
            return True, np.zeros((2160, 3840, 3), dtype=np.uint8)

        def release(self):
            pass

    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        COLOR_BGR2RGB=4,
        VideoCapture=FakeCapture,
        cvtColor=lambda frame, code: frame,
    )
    sys.modules["cv2"] = fake_cv2

    def fake_image_to_tensor(image):
        seen_sizes.append((vision.image_width, vision.image_height, vision.min_pixels))
        return torch.zeros((1, 3, vision.image_height, vision.image_width))

    def fake_videos_forward(frames):
        seq_len = (frames[0].shape[2] // vision.patch_size) * (frames[0].shape[3] // vision.patch_size)
        return torch.zeros((seq_len, 1, 1))

    vision.image_to_tensor = fake_image_to_tensor
    vision.videos_forward = fake_videos_forward
    try:
        segments = Qwen2Vision.video_process(vision, "missing-meta.avi")
    finally:
        if old_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = old_cv2

    assert len(seen_sizes) == 2
    width, height, min_pixels = seen_sizes[0]
    assert width * height <= vision.video_max_pixels
    assert min_pixels <= width * height
    assert segments
    assert vision.image_width == 448
    assert vision.image_height == 448


def test_str_to_ids_resets_multimodal_state():
    vision = make_qwen_vision()
    vision.image_embeds = [object()]
    vision.video_embeds = [object()]
    vision.image_grid_thw = [[1, 4, 4]]
    vision.vision_segments = [{"type": "video", "grid": [1, 4, 4]}]
    vision.deepstack_feature_list = [object()]
    vision.deepstack_embeds = object()

    Qwen2Vision.str_to_ids(vision, "plain text")

    assert vision.image_embeds == []
    assert vision.video_embeds == []
    assert vision.image_grid_thw == []
    assert vision.vision_segments == []
    assert vision.deepstack_feature_list == []
    assert vision.deepstack_embeds is None


def test_video_prompt_requires_video_token_id():
    vision = make_qwen_vision()
    vision.video_pad_id = None

    try:
        Qwen2Vision.str_to_ids(vision, "describe <video>demo.mp4</video> now")
    except RuntimeError as exc:
        assert "video_token_id" in str(exc)
    else:
        raise AssertionError("video prompt without video_token_id should fail")


def test_visual_position_ids_repeat_for_temporal_video_grid():
    vision = make_qwen_vision()
    position_ids = Qwen2Vision.vision_position_ids(vision, torch.tensor([[2, 4, 4]]))

    expected_one_temporal = torch.tensor([
        [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
        [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3],
    ])
    assert position_ids.shape == (2, 32)
    assert torch.equal(position_ids[:, :16], expected_one_temporal)
    assert torch.equal(position_ids[:, 16:], expected_one_temporal)


if __name__ == "__main__":
    test_video_prompt_expands_to_timestamped_video_pads()
    test_video_position_ids_use_video_grid_not_text_positions()
    test_image_prompt_still_uses_image_pads()
    test_video_position_ids_advance_by_largest_axis_for_non_square_grid()
    test_video_sampling_matches_qwen3vl_min_frames_and_linspace()
    test_video_resize_preserves_aspect_and_caps_pixels()
    test_video_resize_uses_total_token_budget_for_long_video()
    test_video_attention_mask_rejects_over_budget_before_allocation()
    test_video_process_uses_first_frame_size_when_metadata_is_missing()
    test_str_to_ids_resets_multimodal_state()
    test_video_prompt_requires_video_token_id()
    test_visual_position_ids_repeat_for_temporal_video_grid()
