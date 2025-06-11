#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a streaming model.

Usage:

(1) Streaming transducer

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
rm sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

./python-api-examples/online-decode-files.py \
  --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-64.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-64.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-64.onnx \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/1.wav \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/8k.wav

(2) Streaming paraformer

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

./python-api-examples/online-decode-files.py \
  --tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
  --paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/1.wav \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/2.wav \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/3.wav \
  ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/8k.wav

(3) Streaming Zipformer2 CTC

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
ls -lh sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13

./python-api-examples/online-decode-files.py \
  --zipformer2-ctc=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
  --tokens=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt \
  ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav \
  ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000001.wav

(4) Streaming Conformer CTC from WeNet

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zh-wenet-wenetspeech.tar.bz2
tar xvf sherpa-onnx-zh-wenet-wenetspeech.tar.bz2
rm sherpa-onnx-zh-wenet-wenetspeech.tar.bz2

./python-api-examples/online-decode-files.py \
  --tokens=./sherpa-onnx-zh-wenet-wenetspeech/tokens.txt \
  --wenet-ctc=./sherpa-onnx-zh-wenet-wenetspeech/model-streaming.onnx \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/0.wav \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/1.wav \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/8k.wav


Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
to download streaming pre-trained models.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sherpa_mnn


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--zipformer2-ctc",
        type=str,
        help="Path to the zipformer2 ctc model",
    )

    parser.add_argument(
        "--paraformer-encoder",
        type=str,
        help="Path to the paraformer encoder model",
    )

    parser.add_argument(
        "--paraformer-decoder",
        type=str,
        help="Path to the paraformer decoder model",
    )

    parser.add_argument(
        "--wenet-ctc",
        type=str,
        help="Path to the wenet ctc model",
    )

    parser.add_argument(
        "--wenet-ctc-chunk-size",
        type=int,
        default=16,
        help="The --chunk-size parameter for streaming WeNet models",
    )

    parser.add_argument(
        "--wenet-ctc-num-left-chunks",
        type=int,
        default=4,
        help="The --num-left-chunks parameter for streaming WeNet models",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--lm",
        type=str,
        default="",
        help="""Used only when --decoding-method is modified_beam_search.
        path of language model.
        """,
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.1,
        help="""Used only when --decoding-method is modified_beam_search.
        scale of language model.
        """,
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, like
        HELLO WORLD
        你好世界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--modeling-unit",
        type=str,
        default="",
        help="""
        The modeling unit of the model, valid values are cjkchar, bpe, cjkchar+bpe.
        Used only when hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--bpe-vocab",
        type=str,
        default="",
        help="""
        The path to the bpe vocabulary, the bpe vocabulary is generated by
        sentencepiece, you can also export the bpe vocabulary through a bpe model
        by `scripts/export_bpe_vocab.py`. Used only when hotwords-file is given
        and modeling-unit is bpe or cjkchar+bpe.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to decode. Each file must be of WAVE"
        "format with a single channel, and each sample has 16-bit, "
        "i.e., int16_t. "
        "The sample rate of the file can be arbitrary and does not need to "
        "be 16 kHz",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


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


def main():
    args = get_args()
    assert_file_exists(args.tokens)

    if args.encoder:
        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        assert not args.paraformer_encoder, args.paraformer_encoder
        assert not args.paraformer_decoder, args.paraformer_decoder

        recognizer = sherpa_mnn.OnlineRecognizer.from_transducer(
            tokens=args.tokens,
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            num_threads=args.num_threads,
            provider=args.provider,
            sample_rate=16000,
            feature_dim=80,
            decoding_method=args.decoding_method,
            max_active_paths=args.max_active_paths,
            lm=args.lm,
            lm_scale=args.lm_scale,
            hotwords_file=args.hotwords_file,
            hotwords_score=args.hotwords_score,
            modeling_unit=args.modeling_unit,
            bpe_vocab=args.bpe_vocab,
            blank_penalty=args.blank_penalty,
        )
    elif args.zipformer2_ctc:
        recognizer = sherpa_mnn.OnlineRecognizer.from_zipformer2_ctc(
            tokens=args.tokens,
            model=args.zipformer2_ctc,
            num_threads=args.num_threads,
            provider=args.provider,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    elif args.paraformer_encoder:
        recognizer = sherpa_mnn.OnlineRecognizer.from_paraformer(
            tokens=args.tokens,
            encoder=args.paraformer_encoder,
            decoder=args.paraformer_decoder,
            num_threads=args.num_threads,
            provider=args.provider,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    elif args.wenet_ctc:
        recognizer = sherpa_mnn.OnlineRecognizer.from_wenet_ctc(
            tokens=args.tokens,
            model=args.wenet_ctc,
            chunk_size=args.wenet_ctc_chunk_size,
            num_left_chunks=args.wenet_ctc_num_left_chunks,
            num_threads=args.num_threads,
            provider=args.provider,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    else:
        raise ValueError("Please provide a model")

    print("Started!")
    start_time = time.time()

    streams = []
    total_duration = 0
    for wave_filename in args.sound_files:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        s = recognizer.create_stream()

        s.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)

        s.input_finished()

        streams.append(s)

    while True:
        ready_list = []
        for s in streams:
            if recognizer.is_ready(s):
                ready_list.append(s)
        if len(ready_list) == 0:
            break
        recognizer.decode_streams(ready_list)
    results = [recognizer.get_result(s) for s in streams]
    end_time = time.time()
    print("Done!")

    for wave_filename, result in zip(args.sound_files, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / total_duration
    print(f"num_threads: {args.num_threads}")
    print(f"decoding_method: {args.decoding_method}")
    print(f"Wave duration: {total_duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    )


if __name__ == "__main__":
    main()
