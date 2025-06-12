# Copyright (c)  2023  Xiaomi Corporation
from pathlib import Path
from typing import List, Optional

from _sherpa_mnn import (
    EndpointConfig,
    FeatureExtractorConfig,
    OnlineLMConfig,
    OnlineModelConfig,
    OnlineParaformerModelConfig,
)
from _sherpa_mnn import OnlineRecognizer as _Recognizer
from _sherpa_mnn import (
    CudaConfig,
    TensorrtConfig,
    ProviderConfig,
    OnlineRecognizerConfig,
    OnlineRecognizerResult,
    OnlineStream,
    OnlineTransducerModelConfig,
    OnlineWenetCtcModelConfig,
    OnlineNeMoCtcModelConfig,
    OnlineZipformer2CtcModelConfig,
    OnlineCtcFstDecoderConfig,
)


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


class OnlineRecognizer(object):
    """A class for streaming speech recognition.

    Please refer to the following files for usages
     - https://github.com/k2-fsa/sherpa-mnn/blob/master/sherpa-mnn/python/tests/test_online_recognizer.py
     - https://github.com/k2-fsa/sherpa-mnn/blob/master/python-api-examples/online-decode-files.py
    """

    @classmethod
    def from_transducer(
        cls,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
        dither: float = 0.0,
        normalize_samples: bool = True,
        snip_edges: bool = False,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        hotwords_score: float = 1.5,
        blank_penalty: float = 0.0,
        hotwords_file: str = "",
        model_type: str = "",
        modeling_unit: str = "cjkchar",
        bpe_vocab: str = "",
        lm: str = "",
        lm_scale: float = 0.1,
        lm_shallow_fusion: bool = True,
        temperature_scale: float = 2.0,
        debug: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        provider: str = "cpu",
        device: int = 0,
        cudnn_conv_algo_search: int = 1,
        trt_max_workspace_size: int = 2147483647,
        trt_max_partition_iterations: int = 10,
        trt_min_subgraph_size: int = 5,
        trt_fp16_enable: bool = True,
        trt_detailed_build_log: bool = False,
        trt_engine_cache_enable: bool = True,
        trt_timing_cache_enable: bool = True,
        trt_engine_cache_path: str ="",
        trt_timing_cache_path: str ="",
        trt_dump_subgraphs: bool = False,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>`_
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
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          low_freq:
            Low cutoff frequency for mel bins in feature extraction.
          high_freq:
            High cutoff frequency for mel bins in feature extraction
            (if <= 0, offset from Nyquist)
          dither:
            Dithering constant (0.0 means no dither).
            By default the audio samples are in range [-1,+1],
            so dithering constant 0.00003 is a good value,
            equivalent to the default 1.0 from kaldi
          normalize_samples:
            True for +/- 1.0 range of audio samples (default, zipformer feats),
            False for +/- 32k samples (ebranchformer features).
          snip_edges:
            handling of end of audio signal in kaldi feature extraction.
            If true, end effects will be handled by outputting only frames that
            completely fit in the file, and the number of frames depends on the
            frame-length.  If false, the number of frames depends only on the
            frame-shift, and we reflect the data at the ends.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            Valid values are greedy_search, modified_beam_search.
          max_active_paths:
            Use only when decoding_method is modified_beam_search. It specifies
            the maximum number of active paths during beam search.
          blank_penalty:
            The penalty applied on blank symbol during decoding.
          hotwords_file:
            The file containing hotwords, one words/phrases per line, and for each
            phrase the bpe/cjkchar are separated by a space.
          hotwords_score:
            The hotword score of each token for biasing word/phrase. Used only if
            hotwords_file is given with modified_beam_search as decoding method.
          temperature_scale:
            Temperature scaling for output symbol confidence estiamation.
            It affects only confidence values, the decoding uses the original
            logits without temperature.
          model_type:
            Online transducer model type. Valid values are: conformer, lstm,
            zipformer, zipformer2. All other values lead to loading the model twice.
          modeling_unit:
            The modeling unit of the model, commonly used units are bpe, cjkchar,
            cjkchar+bpe, etc. Currently, it is needed only when hotwords are
            provided, we need it to encode the hotwords into token sequence.
          bpe_vocab:
            The vocabulary generated by google's sentencepiece program.
            It is a file has two columns, one is the token, the other is
            the log probability, you can get it from the directory where
            your bpe model is generated. Only used when hotwords provided
            and the modeling unit is bpe or cjkchar+bpe.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          device:
            onnxruntime cuda device index.
          cudnn_conv_algo_search:
            onxrt CuDNN convolution search algorithm selection. CUDA EP
          trt_max_workspace_size:
            Set TensorRT EP GPU memory usage limit. TensorRT EP
          trt_max_partition_iterations:
            Limit partitioning iterations for model conversion. TensorRT EP
          trt_min_subgraph_size:
            Set minimum size for subgraphs in partitioning. TensorRT EP
          trt_fp16_enable: bool = True,
            Enable FP16 precision for faster performance. TensorRT EP
          trt_detailed_build_log: bool = False,
            Enable detailed logging of build steps. TensorRT EP
          trt_engine_cache_enable: bool = True,
            Enable caching of TensorRT engines. TensorRT EP
          trt_timing_cache_enable: bool = True,
            "Enable use of timing cache to speed up builds." TensorRT EP
          trt_engine_cache_path: str ="",
            "Set path to store cached TensorRT engines." TensorRT EP
          trt_timing_cache_path: str ="",
            "Set path for storing timing cache." TensorRT EP
          trt_dump_subgraphs: bool = False,
            "Dump optimized subgraphs for debugging." TensorRT EP
        """
        self = cls.__new__(cls)
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

        cuda_config = CudaConfig(
          cudnn_conv_algo_search=cudnn_conv_algo_search,
        )

        trt_config = TensorrtConfig(
          trt_max_workspace_size=trt_max_workspace_size,
          trt_max_partition_iterations=trt_max_partition_iterations,
          trt_min_subgraph_size=trt_min_subgraph_size,
          trt_fp16_enable=trt_fp16_enable,
          trt_detailed_build_log=trt_detailed_build_log,
          trt_engine_cache_enable=trt_engine_cache_enable,
          trt_timing_cache_enable=trt_timing_cache_enable,
          trt_engine_cache_path=trt_engine_cache_path,
          trt_timing_cache_path=trt_timing_cache_path,
          trt_dump_subgraphs=trt_dump_subgraphs,
        )

        provider_config = ProviderConfig(
          trt_config=trt_config,
          cuda_config=cuda_config,
          provider=provider,
          device=device,
        )

        model_config = OnlineModelConfig(
            transducer=transducer_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
            model_type=model_type,
            modeling_unit=modeling_unit,
            bpe_vocab=bpe_vocab,
            debug=debug,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            normalize_samples=normalize_samples,
            snip_edges=snip_edges,
            feature_dim=feature_dim,
            low_freq=low_freq,
            high_freq=high_freq,
            dither=dither,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        if len(hotwords_file) > 0 and decoding_method != "modified_beam_search":
            raise ValueError(
                "Please use --decoding-method=modified_beam_search when using "
                f"--hotwords-file. Currently given: {decoding_method}"
            )

        if lm and decoding_method != "modified_beam_search":
            raise ValueError(
                "Please use --decoding-method=modified_beam_search when using "
                f"--lm. Currently given: {decoding_method}"
            )

        lm_config = OnlineLMConfig(
            model=lm,
            scale=lm_scale,
            shallow_fusion=lm_shallow_fusion,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            lm_config=lm_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
            hotwords_score=hotwords_score,
            hotwords_file=hotwords_file,
            blank_penalty=blank_penalty,
            temperature_scale=temperature_scale,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_paraformer(
        cls,
        tokens: str,
        encoder: str,
        decoder: str,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        debug: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        device: int = 0,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>`_
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
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            The only valid value is greedy_search.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
          device:
            onnxruntime cuda device index.
        """
        self = cls.__new__(cls)
        _assert_file_exists(tokens)
        _assert_file_exists(encoder)
        _assert_file_exists(decoder)

        assert num_threads > 0, num_threads

        paraformer_config = OnlineParaformerModelConfig(
            encoder=encoder,
            decoder=decoder,
        )

        provider_config = ProviderConfig(
          provider=provider,
          device=device,
        )

        model_config = OnlineModelConfig(
            paraformer=paraformer_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
            model_type="paraformer",
            debug=debug,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_zipformer2_ctc(
        cls,
        tokens: str,
        model: str,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        ctc_graph: str = "",
        ctc_max_active: int = 3000,
        provider: str = "cpu",
        debug: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        device: int = 0,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          model:
            Path to ``model.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            The only valid value is greedy_search.
          ctc_graph:
            If not empty, decoding_method is ignored. It contains the path to
            H.fst, HL.fst, or HLG.fst
          ctc_max_active:
            Used only when ctc_graph is not empty. It specifies the maximum
            active paths at a time.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
          device:
            onnxruntime cuda device index.
        """
        self = cls.__new__(cls)
        _assert_file_exists(tokens)
        _assert_file_exists(model)

        assert num_threads > 0, num_threads

        zipformer2_ctc_config = OnlineZipformer2CtcModelConfig(model=model)

        provider_config = ProviderConfig(
          provider=provider,
          device=device,
        )

        model_config = OnlineModelConfig(
            zipformer2_ctc=zipformer2_ctc_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
            debug=debug,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        ctc_fst_decoder_config = OnlineCtcFstDecoderConfig(
            graph=ctc_graph,
            max_active=ctc_max_active,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            endpoint_config=endpoint_config,
            ctc_fst_decoder_config=ctc_fst_decoder_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_nemo_ctc(
        cls,
        tokens: str,
        model: str,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        debug: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        device: int = 0,
    ):
        """
        Please refer to
        `<https://github.com/k2-fsa/sherpa-mnn/releases/tag/asr-models>`_
        to download pre-trained models.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          model:
            Path to ``model.onnx``.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            The only valid value is greedy_search.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          debug:
            True to show meta data in the model.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
          device:
            onnxruntime cuda device index.
        """
        self = cls.__new__(cls)
        _assert_file_exists(tokens)
        _assert_file_exists(model)

        assert num_threads > 0, num_threads

        nemo_ctc_config = OnlineNeMoCtcModelConfig(
            model=model,
        )

        provider_config = ProviderConfig(
          provider=provider,
          device=device,
        )

        model_config = OnlineModelConfig(
            nemo_ctc=nemo_ctc_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
            debug=debug,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    @classmethod
    def from_wenet_ctc(
        cls,
        tokens: str,
        model: str,
        chunk_size: int = 16,
        num_left_chunks: int = 4,
        num_threads: int = 2,
        sample_rate: float = 16000,
        feature_dim: int = 80,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        debug: bool = False,
        rule_fsts: str = "",
        rule_fars: str = "",
        device: int = 0,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/wenet/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id

          model:
            Path to ``model.onnx``.
          chunk_size:
            The --chunk-size parameter from WeNet.
          num_left_chunks:
            The --num-left-chunks parameter from WeNet.
          num_threads:
            Number of threads for neural network computation.
          sample_rate:
            Sample rate of the training data used to train the model.
          feature_dim:
            Dimension of the feature used to train the model.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          decoding_method:
            The only valid value is greedy_search.
          provider:
            onnxruntime execution providers. Valid values are: cpu, cuda, coreml.
          rule_fsts:
            If not empty, it specifies fsts for inverse text normalization.
            If there are multiple fsts, they are separated by a comma.
          rule_fars:
            If not empty, it specifies fst archives for inverse text normalization.
            If there are multiple archives, they are separated by a comma.
          device:
            onnxruntime cuda device index.
        """
        self = cls.__new__(cls)
        _assert_file_exists(tokens)
        _assert_file_exists(model)

        assert num_threads > 0, num_threads

        wenet_ctc_config = OnlineWenetCtcModelConfig(
            model=model,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks,
        )

        provider_config = ProviderConfig(
          provider=provider,
          device=device,
        )

        model_config = OnlineModelConfig(
            wenet_ctc=wenet_ctc_config,
            tokens=tokens,
            num_threads=num_threads,
            provider_config=provider_config,
            debug=debug,
        )

        feat_config = FeatureExtractorConfig(
            sampling_rate=sample_rate,
            feature_dim=feature_dim,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        recognizer_config = OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
            decoding_method=decoding_method,
            rule_fsts=rule_fsts,
            rule_fars=rule_fars,
        )

        self.recognizer = _Recognizer(recognizer_config)
        self.config = recognizer_config
        return self

    def create_stream(self, hotwords: Optional[str] = None):
        if hotwords is None:
            return self.recognizer.create_stream()
        else:
            return self.recognizer.create_stream(hotwords)

    def decode_stream(self, s: OnlineStream):
        self.recognizer.decode_stream(s)

    def decode_streams(self, ss: List[OnlineStream]):
        self.recognizer.decode_streams(ss)

    def is_ready(self, s: OnlineStream) -> bool:
        return self.recognizer.is_ready(s)

    def get_result_all(self, s: OnlineStream) -> OnlineRecognizerResult:
        return self.recognizer.get_result(s)

    def get_result(self, s: OnlineStream) -> str:
        return self.recognizer.get_result(s).text.strip()

    def get_result_as_json_string(self, s: OnlineStream) -> str:
        return self.recognizer.get_result(s).as_json_string()

    def tokens(self, s: OnlineStream) -> List[str]:
        return self.recognizer.get_result(s).tokens

    def timestamps(self, s: OnlineStream) -> List[float]:
        return self.recognizer.get_result(s).timestamps

    def start_time(self, s: OnlineStream) -> float:
        return self.recognizer.get_result(s).start_time

    def ys_probs(self, s: OnlineStream) -> List[float]:
        return self.recognizer.get_result(s).ys_probs

    def lm_probs(self, s: OnlineStream) -> List[float]:
        return self.recognizer.get_result(s).lm_probs

    def context_scores(self, s: OnlineStream) -> List[float]:
        return self.recognizer.get_result(s).context_scores

    def is_endpoint(self, s: OnlineStream) -> bool:
        return self.recognizer.is_endpoint(s)

    def reset(self, s: OnlineStream) -> bool:
        return self.recognizer.reset(s)
