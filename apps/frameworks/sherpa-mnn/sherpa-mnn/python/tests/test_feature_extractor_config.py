# sherpa-mnn/python/tests/test_feature_extractor_config.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_feature_extractor_config_py

import unittest

import _sherpa_mnn


class TestFeatureExtractorConfig(unittest.TestCase):
    def test_default_constructor(self):
        config = _sherpa_mnn.FeatureExtractorConfig()
        assert config.sampling_rate == 16000, config.sampling_rate
        assert config.feature_dim == 80, config.feature_dim
        print(config)

    def test_constructor(self):
        config = _sherpa_mnn.FeatureExtractorConfig(sampling_rate=8000, feature_dim=40)
        assert config.sampling_rate == 8000, config.sampling_rate
        assert config.feature_dim == 40, config.feature_dim
        print(config)


if __name__ == "__main__":
    unittest.main()
