# sherpa-mnn/python/tests/test_online_transducer_model_config.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_online_transducer_model_config_py

import unittest

import _sherpa_mnn


class TestOnlineTransducerModelConfig(unittest.TestCase):
    def test_constructor(self):
        config = _sherpa_mnn.OnlineTransducerModelConfig(
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            joiner="joiner.onnx",
        )
        assert config.encoder == "encoder.onnx", config.encoder
        assert config.decoder == "decoder.onnx", config.decoder
        assert config.joiner == "joiner.onnx", config.joiner
        print(config)


if __name__ == "__main__":
    unittest.main()
