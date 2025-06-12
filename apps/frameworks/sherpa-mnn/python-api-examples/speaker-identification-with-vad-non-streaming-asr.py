#!/usr/bin/env python3

"""
This script shows how to use Python APIs for speaker identification with
a microphone, a VAD model, and a non-streaming ASR model.

Please see also ./generate-subtitles.py

Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Download the VAD model
Please visit
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

(4) Please refer to ./generate-subtitles.py
to download a non-streaming ASR model.

(5) Run this script

Assume the filename of the text file is speaker.txt.

python3 ./python-api-examples/speaker-identification-with-vad-non-streaming-asr.py \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --speaker-file ./speaker.txt \
  --model ./wespeaker_zh_cnceleb_resnet34.onnx
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sherpa_mnn
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

g_sample_rate = 16000


def register_non_streaming_asr_model_args(parser):
    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the CTC model.onnx from WeNet",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Valid values are greedy_search and modified_beam_search.
        modified_beam_search is valid only for transducer models.
        """,
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to sense voice model",
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    register_non_streaming_asr_model_args(parser)

    parser.add_argument(
        "--speaker-file",
        type=str,
        required=True,
        help="""Path to the speaker file. Read the help doc at the beginning of this
        file for the format.""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the speaker embedding model file.",
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    parser.add_argument("--threshold", type=float, default=0.6)

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_mnn.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_mnn.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.paraformer:
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_mnn.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=g_sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.wenet_ctc:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.wenet_ctc)

        recognizer = sherpa_mnn.OfflineRecognizer.from_wenet_ctc(
            model=args.wenet_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)

        recognizer = sherpa_mnn.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
        )
    elif args.sense_voice:
        assert_file_exists(args.sense_voice)
        recognizer = sherpa_mnn.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
        )
    else:
        raise ValueError("Please specify at least one model")

    return recognizer


def load_speaker_embedding_model(args):
    config = sherpa_mnn.SpeakerEmbeddingExtractorConfig(
        model=args.model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_mnn.SpeakerEmbeddingExtractor(config)
    return extractor


def load_speaker_file(args) -> Dict[str, List[str]]:
    if not Path(args.speaker_file).is_file():
        raise ValueError(f"--speaker-file {args.speaker_file} does not exist")

    ans = defaultdict(list)
    with open(args.speaker_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, filename = fields
            ans[speaker_name].append(filename)
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_speaker_embedding(
    filenames: List[str],
    extractor: sherpa_mnn.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    assert len(filenames) > 0, "filenames is empty"

    ans = None
    for filename in filenames:
        print(f"processing {filename}")
        samples, sample_rate = load_audio(filename)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        if ans is None:
            ans = embedding
        else:
            ans += embedding

    return ans / len(filenames)


def main():
    args = get_args()
    print(args)
    recognizer = create_recognizer(args)
    extractor = load_speaker_embedding_model(args)
    speaker_file = load_speaker_file(args)

    manager = sherpa_mnn.SpeakerEmbeddingManager(extractor.dim)
    for name, filename_list in speaker_file.items():
        embedding = compute_speaker_embedding(
            filenames=filename_list,
            extractor=extractor,
        )
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")

    vad_config = sherpa_mnn.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.sample_rate = g_sample_rate
    if not vad_config.validate():
        raise ValueError("Errors in vad config")

    window_size = vad_config.silero_vad.window_size

    vad = sherpa_mnn.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    samples_per_read = int(0.1 * g_sample_rate)  # 0.1 second = 100 ms

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Started! Please speak")

    idx = 0
    buffer = []
    with sd.InputStream(channels=1, dtype="float32", samplerate=g_sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                if len(vad.front.samples) < 0.5 * g_sample_rate:
                    # this segment is too short, skip it
                    vad.pop()
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(
                    sample_rate=g_sample_rate, waveform=vad.front.samples
                )
                stream.input_finished()

                embedding = extractor.compute(stream)
                embedding = np.array(embedding)
                name = manager.search(embedding, threshold=args.threshold)
                if not name:
                    name = "unknown"

                # Now for non-streaming ASR
                asr_stream = recognizer.create_stream()
                asr_stream.accept_waveform(
                    sample_rate=g_sample_rate, waveform=vad.front.samples
                )
                recognizer.decode_stream(asr_stream)
                text = asr_stream.result.text

                vad.pop()

                print(f"\r{idx}-{name}: {text}")
                idx += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
