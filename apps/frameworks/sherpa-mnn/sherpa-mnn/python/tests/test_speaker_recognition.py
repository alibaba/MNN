# sherpa-mnn/python/tests/test_speaker_recognition.py
#
# Copyright (c)  2024  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_speaker_recognition_py

import unittest
import wave
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_mnn

d = "/tmp/sr-models"


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def load_speaker_embedding_model(model_filename):
    config = sherpa_mnn.SpeakerEmbeddingExtractorConfig(
        model=model_filename,
        num_threads=1,
        debug=True,
        provider="cpu",
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_mnn.SpeakerEmbeddingExtractor(config)
    return extractor


def test_zh_models(model_filename: str):
    model_filename = str(model_filename)
    if "en" in model_filename:
        print(f"skip {model_filename}")
        return
    extractor = load_speaker_embedding_model(model_filename)
    filenames = [
        "leijun-sr-1",
        "leijun-sr-2",
        "fangjun-sr-1",
        "fangjun-sr-2",
        "fangjun-sr-3",
    ]
    tmp = defaultdict(list)
    for filename in filenames:
        print(filename)
        name = filename.split("-", maxsplit=1)[0]
        data, sample_rate = read_wave(f"/tmp/sr-models/sr-data/enroll/{filename}.wav")
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=data)
        stream.input_finished()
        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        tmp[name].append(embedding)

    manager = sherpa_mnn.SpeakerEmbeddingManager(extractor.dim)
    for name, embedding_list in tmp.items():
        print(name, len(embedding_list))
        embedding = sum(embedding_list) / len(embedding_list)
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")

    filenames = [
        "leijun-test-sr-1",
        "leijun-test-sr-2",
        "leijun-test-sr-3",
        "fangjun-test-sr-1",
        "fangjun-test-sr-2",
    ]
    for filename in filenames:
        name = filename.split("-", maxsplit=1)[0]
        data, sample_rate = read_wave(f"/tmp/sr-models/sr-data/test/{filename}.wav")
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=data)
        stream.input_finished()
        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        status = manager.verify(name, embedding, threshold=0.5)
        if not status:
            raise RuntimeError(f"Failed to verify {name} with wave {filename}.wav")

        ans = manager.search(embedding, threshold=0.5)
        assert ans == name, (name, ans)


def test_en_and_zh_models(model_filename: str):
    model_filename = str(model_filename)
    extractor = load_speaker_embedding_model(model_filename)
    manager = sherpa_mnn.SpeakerEmbeddingManager(extractor.dim)

    filenames = [
        "speaker1_a_cn_16k",
        "speaker2_a_cn_16k",
        "speaker1_a_en_16k",
        "speaker2_a_en_16k",
    ]
    is_en = "en" in model_filename
    for filename in filenames:
        if is_en and "cn" in filename:
            continue

        if not is_en and "en" in filename:
            continue

        name = filename.rsplit("_", maxsplit=1)[0]
        data, sample_rate = read_wave(
            f"/tmp/sr-models/sr-data/test/3d-speaker/{filename}.wav"
        )
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=data)
        stream.input_finished()
        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)

        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")

    filenames = [
        "speaker1_b_cn_16k",
        "speaker1_b_en_16k",
    ]
    for filename in filenames:
        if is_en and "cn" in filename:
            continue

        if not is_en and "en" in filename:
            continue
        print(filename)
        name = filename.rsplit("_", maxsplit=1)[0]
        name = name.replace("b_cn", "a_cn")
        name = name.replace("b_en", "a_en")
        print(name)

        data, sample_rate = read_wave(
            f"/tmp/sr-models/sr-data/test/3d-speaker/{filename}.wav"
        )
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=data)
        stream.input_finished()
        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        status = manager.verify(name, embedding, threshold=0.5)
        if not status:
            raise RuntimeError(
                f"Failed to verify {name} with wave {filename}.wav. model: {model_filename}"
            )

        ans = manager.search(embedding, threshold=0.5)
        assert ans == name, (name, ans)


class TestSpeakerRecognition(unittest.TestCase):
    def test_wespeaker_models(self):
        model_dir = Path(d) / "wespeaker"
        if not model_dir.is_dir():
            print(f"{model_dir} does not exist - skip it")
            return
        for filename in model_dir.glob("*.onnx"):
            print(filename)
            test_zh_models(filename)
            test_en_and_zh_models(filename)

    def _test_3dpeaker_models(self):
        model_dir = Path(d) / "3dspeaker"
        if not model_dir.is_dir():
            print(f"{model_dir} does not exist - skip it")
            return
        for filename in model_dir.glob("*.onnx"):
            print(filename)
            test_en_and_zh_models(filename)

    def test_nemo_models(self):
        model_dir = Path(d) / "nemo"
        if not model_dir.is_dir():
            print(f"{model_dir} does not exist - skip it")
            return
        for filename in model_dir.glob("*.onnx"):
            print(filename)
            test_en_and_zh_models(filename)


if __name__ == "__main__":
    unittest.main()
