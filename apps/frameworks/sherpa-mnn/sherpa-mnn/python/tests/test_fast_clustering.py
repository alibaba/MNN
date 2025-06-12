# sherpa-mnn/python/tests/test_fast_clustering.py
#
# Copyright (c)  2024  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_fast_clustering_py
import unittest

import sherpa_mnn
import numpy as np
from pathlib import Path
from typing import Tuple

import soundfile as sf


def load_audio(filename: str) -> np.ndarray:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    assert sample_rate == 16000, f"Expect sample_rate 16000. Given: {sample_rate}"
    return samples


class TestFastClustering(unittest.TestCase):
    def test_construct_by_num_clusters(self):
        config = sherpa_mnn.FastClusteringConfig(num_clusters=4)
        assert config.validate() is True

        print(config)

        clustering = sherpa_mnn.FastClustering(config)
        features = np.array(
            [
                [0.2, 0.3],  # cluster 0
                [0.3, -0.4],  # cluster 1
                [-0.1, -0.2],  # cluster 2
                [-0.3, -0.5],  # cluster 2
                [0.1, -0.2],  # cluster 1
                [0.1, 0.2],  # cluster 0
                [-0.8, 1.9],  # cluster 3
                [-0.4, -0.6],  # cluster 2
                [-0.7, 0.9],  # cluster 3
            ]
        )
        labels = clustering(features)
        assert isinstance(labels, list)
        assert len(labels) == features.shape[0]

        expected = [0, 1, 2, 2, 1, 0, 3, 2, 3]
        assert labels == expected, (labels, expected)

    def test_construct_by_threshold(self):
        config = sherpa_mnn.FastClusteringConfig(threshold=0.2)
        assert config.validate() is True

        print(config)

        clustering = sherpa_mnn.FastClustering(config)
        features = np.array(
            [
                [0.2, 0.3],  # cluster 0
                [0.3, -0.4],  # cluster 1
                [-0.1, -0.2],  # cluster 2
                [-0.3, -0.5],  # cluster 2
                [0.1, -0.2],  # cluster 1
                [0.1, 0.2],  # cluster 0
                [-0.8, 1.9],  # cluster 3
                [-0.4, -0.6],  # cluster 2
                [-0.7, 0.9],  # cluster 3
            ]
        )
        labels = clustering(features)
        assert isinstance(labels, list)
        assert len(labels) == features.shape[0]

        expected = [0, 1, 2, 2, 1, 0, 3, 2, 3]
        assert labels == expected, (labels, expected)

    def test_cluster_speaker_embeddings(self):
        d = Path("/tmp/test-cluster")

        # Please download the onnx file from
        # https://github.com/k2-fsa/sherpa-mnn/releases/tag/speaker-recongition-models
        model_file = d / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

        if not model_file.exists():
            print(f"skip test since {model_file} does not exist")
            return

        # Please download the test wave files from
        # https://github.com/csukuangfj/sr-data
        wave_dir = d / "sr-data"
        if not wave_dir.is_dir():
            print(f"skip test since {wave_dir} does not exist")
            return

        wave_files = [
            "enroll/fangjun-sr-1.wav",  # cluster 0
            "enroll/fangjun-sr-2.wav",  # cluster 0
            "enroll/fangjun-sr-3.wav",  # cluster 0
            "enroll/leijun-sr-1.wav",  # cluster 1
            "enroll/leijun-sr-2.wav",  # cluster 1
            "enroll/liudehua-sr-1.wav",  # cluster 2
            "enroll/liudehua-sr-2.wav",  # cluster 2
            "test/fangjun-test-sr-1.wav",  # cluster 0
            "test/fangjun-test-sr-2.wav",  # cluster 0
            "test/leijun-test-sr-1.wav",  # cluster 1
            "test/leijun-test-sr-2.wav",  # cluster 1
            "test/leijun-test-sr-3.wav",  # cluster 1
            "test/liudehua-test-sr-1.wav",  # cluster 2
            "test/liudehua-test-sr-2.wav",  # cluster 2
        ]
        for w in wave_files:
            f = d / "sr-data" / w
            if not f.is_file():
                print(f"skip testing since {f} does not exist")
                return

        extractor_config = sherpa_mnn.SpeakerEmbeddingExtractorConfig(
            model=str(model_file),
            num_threads=1,
            debug=0,
        )
        if not extractor_config.validate():
            raise ValueError(f"Invalid extractor config. {config}")

        extractor = sherpa_mnn.SpeakerEmbeddingExtractor(extractor_config)

        features = []

        for w in wave_files:
            f = d / "sr-data" / w
            audio = load_audio(str(f))
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=audio)
            stream.input_finished()

            assert extractor.is_ready(stream)
            embedding = extractor.compute(stream)
            embedding = np.array(embedding)
            features.append(embedding)
        features = np.array(features)

        config = sherpa_mnn.FastClusteringConfig(num_clusters=3)
        #  config = sherpa_mnn.FastClusteringConfig(threshold=0.5)
        clustering = sherpa_mnn.FastClustering(config)
        labels = clustering(features)

        expected = [0, 0, 0, 1, 1, 2, 2]
        expected += [0, 0, 1, 1, 1, 2, 2]

        assert labels == expected, (labels, expected)


if __name__ == "__main__":
    unittest.main()
