# Copyright (c)  2023  Xiaomi Corporation

from pathlib import Path
from typing import List, Optional

from _sherpa_mnn import (
    FeatureExtractorConfig,
    KeywordSpotterConfig,
    OnlineModelConfig,
    OnlineTransducerModelConfig,
    OnlineStream,
    ProviderConfig,
)

from _sherpa_mnn import KeywordSpotter as _KeywordSpotter


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class KeywordSpotter(object):
    """A class for keyword spotting.

    Please refer to the following files for usages
     - https://github.com/k2-fsa/sherpa-mnn/blob/master/python-api-examples/keyword-spotter.py
     - https://github.com/k2-fsa/sherpa-mnn/blob/master/python-api-examples/keyword-spotter-from-microphone.py
    """

    def __init__(
        self,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        keywords_file: str,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
        provider: str = "cpu",
        device: int = 0,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          encoder:
            Path to ``encoder.onnx``.
          decoder:
            Path to ``decoder.onnx``.
          joiner:
            Path to ``joiner.onnx``.
          keywords_file:
            The file containing keywords, one word/phrase per line, and for each
            phrase the bpe/cjkchar/pinyin are separated by a space.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          max_active_paths:
            Use only when decoding_method is modified_beam_search. It specifies
            the maximum number of active paths during beam search.
          keywords_score:
            The boosting score of each token for keywords. The larger the easier to
            survive beam search.
          keywords_threshold:
            The trigger threshold (i.e. probability) of the keyword. The larger the
            harder to trigger.
          num_trailing_blanks:
            The number of trailing blanks a keyword should be followed. Setting
            to a larger value (e.g. 8) when your keywords has overlapping tokens
            between each other.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          device:
            onnxruntime cuda device index.
        """
        _assert_file_exists(tokens)
        _assert_file_exists(encoder)
        _assert_file_exists(decoder)
        _assert_file_exists(joiner)

        assert num_threads > 0, num_threads

        transducer_config = OnlineTransducerModelConfig(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
        )

        provider_config = ProviderConfig(
            provider=provider,
            device=device,
        )

        model_config = OnlineModelConfig(
            transducer=transducer_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        keywords_spotter_config = KeywordSpotterConfig(
            feat_config=feat_config,
            model_config=model_config,
            max_active_paths=max_active_paths,
            num_trailing_blanks=num_trailing_blanks,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
            keywords_file=keywords_file,
        )
        self.keyword_spotter = _KeywordSpotter(keywords_spotter_config)

    def reset_stream(self, s: OnlineStream):
        self.keyword_spotter.reset(s)

    def create_stream(self, keywords: Optional[str] = None):
        if keywords is None:
            return self.keyword_spotter.create_stream()
        else:
            return self.keyword_spotter.create_stream(keywords)

    def decode_stream(self, s: OnlineStream):
        self.keyword_spotter.decode_stream(s)

    def decode_streams(self, ss: List[OnlineStream]):
        self.keyword_spotter.decode_streams(ss)

    def is_ready(self, s: OnlineStream) -> bool:
        return self.keyword_spotter.is_ready(s)

    def get_result(self, s: OnlineStream) -> str:
        return self.keyword_spotter.get_result(s).keyword.strip()

    def tokens(self, s: OnlineStream) -> List[str]:
        return self.keyword_spotter.get_result(s).tokens

    def timestamps(self, s: OnlineStream) -> List[float]:
        return self.keyword_spotter.get_result(s).timestamps
