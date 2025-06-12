#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API to generate audio
from text, i.e., text-to-speech.

Different from ./offline-tts.py, this file plays back the generated audio
while the model is still generating.

Usage:

Example (1/7)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
 --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
 --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
 --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
 --output-filename=./generated.wav \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

Example (2/7)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
tar xvf vits-zh-aishell3.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
 --vits-model=./vits-icefall-zh-aishell3/model.onnx \
 --vits-lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
 --vits-tokens=./vits-icefall-zh-aishell3/tokens.txt \
 --tts-rule-fsts='./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst' \
 --sid=21 \
 --output-filename=./liubei-21.wav \
 "勿以恶小而为之，勿以善小而不为。惟贤惟德，能服于人。122334"

Example (3/7)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
rm sherpa-onnx-vits-zh-ll.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
 --vits-model=./sherpa-onnx-vits-zh-ll/model.onnx \
 --vits-lexicon=./sherpa-onnx-vits-zh-ll/lexicon.txt \
 --vits-tokens=./sherpa-onnx-vits-zh-ll/tokens.txt \
 --tts-rule-fsts=./sherpa-onnx-vits-zh-ll/phone.fst,./sherpa-onnx-vits-zh-ll/date.fst,./sherpa-onnx-vits-zh-ll/number.fst \
 --vits-dict-dir=./sherpa-onnx-vits-zh-ll/dict \
 --sid=2 \
 --output-filename=./test-2.wav \
 "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。2024年5月11号，拨打110或者18920240511。123456块钱。"

Example (4/7)

curl -O -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

python3 ./python-api-examples/offline-tts-play.py \
 --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
 --matcha-vocoder=./hifigan_v2.onnx \
 --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
 --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
 --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
 --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
 --output-filename=./test-matcha.wav \
 "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"

Example (5/7)

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

python3 ./python-api-examples/offline-tts-play.py \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
  --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
  --output-filename=./test-matcha-ljspeech-en.wav \
  --num-threads=2 \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

Example (6/7)

(This version of kokoro supports only English)

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

python3 ./python-api-examples/offline-tts.py \
  --debug=1 \
  --kokoro-model=./kokoro-en-v0_19/model.onnx \
  --kokoro-voices=./kokoro-en-v0_19/voices.bin \
  --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
  --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
  --num-threads=2 \
  --sid=10 \
  --output-filename="./kokoro-10.wav" \
  "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be  a statesman, a businessman, an official, or a scholar."

Example (7/7)

(This version of kokoro supports English, Chinese, etc.)

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xf kokoro-multi-lang-v1_0.tar.bz2
rm kokoro-multi-lang-v1_0.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
  --debug=1 \
  --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
  --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
  --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
  --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
  --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
  --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
  --num-threads=2 \
  --sid=18 \
  --output-filename="./kokoro-18-zh-en.wav" \
  "中英文语音合成测试。This is generated by next generation Kaldi using Kokoro without Misaki. 你觉得中英文说的如何呢？"

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/index.html
for details.
"""

import argparse
import logging
import queue
import sys
import threading
import time

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


def add_vits_args(parser):
    parser.add_argument(
        "--vits-model",
        type=str,
        default="",
        help="Path to vits model.onnx",
    )

    parser.add_argument(
        "--vits-lexicon",
        type=str,
        default="",
        help="Path to lexicon.txt",
    )

    parser.add_argument(
        "--vits-tokens",
        type=str,
        default="",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--vits-data-dir",
        type=str,
        default="",
        help="""Path to the dict directory of espeak-ng. If it is specified,
        --vits-lexicon and --vits-tokens are ignored""",
    )

    parser.add_argument(
        "--vits-dict-dir",
        type=str,
        default="",
        help="Path to the dict directory for models using jieba",
    )


def add_matcha_args(parser):
    parser.add_argument(
        "--matcha-acoustic-model",
        type=str,
        default="",
        help="Path to model.onnx for matcha",
    )

    parser.add_argument(
        "--matcha-vocoder",
        type=str,
        default="",
        help="Path to vocoder for matcha",
    )

    parser.add_argument(
        "--matcha-lexicon",
        type=str,
        default="",
        help="Path to lexicon.txt for matcha",
    )

    parser.add_argument(
        "--matcha-tokens",
        type=str,
        default="",
        help="Path to tokens.txt for matcha",
    )

    parser.add_argument(
        "--matcha-data-dir",
        type=str,
        default="",
        help="""Path to the dict directory of espeak-ng. If it is specified,
        --matcha-lexicon and --matcha-tokens are ignored""",
    )

    parser.add_argument(
        "--matcha-dict-dir",
        type=str,
        default="",
        help="Path to the dict directory for models using jieba",
    )


def add_kokoro_args(parser):
    parser.add_argument(
        "--kokoro-model",
        type=str,
        default="",
        help="Path to model.onnx for kokoro",
    )

    parser.add_argument(
        "--kokoro-voices",
        type=str,
        default="",
        help="Path to voices.bin for kokoro",
    )

    parser.add_argument(
        "--kokoro-tokens",
        type=str,
        default="",
        help="Path to tokens.txt for kokoro",
    )

    parser.add_argument(
        "--kokoro-data-dir",
        type=str,
        default="",
        help="Path to the dict directory of espeak-ng.",
    )

    parser.add_argument(
        "--kokoro-dict-dir",
        type=str,
        default="",
        help="Path to the dict directory for models using jieba. Needed only by multilingual kokoro",
    )

    parser.add_argument(
        "--kokoro-lexicon",
        type=str,
        default="",
        help="Path to lexicon.txt for kokoro. Needed only by multilingual kokoro",
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_vits_args(parser)
    add_matcha_args(parser)
    add_kokoro_args(parser)

    parser.add_argument(
        "--tts-rule-fsts",
        type=str,
        default="",
        help="Path to rule.fst",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="./generated.wav",
        help="Path to save generated wave",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
        help="""Speaker ID. Used only for multi-speaker models, e.g.
        models trained using the VCTK dataset. Not used for single-speaker
        models, e.g., models trained using the LJ speech dataset.
        """,
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
        help="valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed. Larger->faster; smaller->slower",
    )

    parser.add_argument(
        "text",
        type=str,
        help="The input text to generate audio for",
    )

    return parser.parse_args()


# buffer saves audio samples to be played
buffer = queue.Queue()

# started is set to True once generated_audio_callback is called.
started = False

# stopped is set to True once all the text has been processed
stopped = False

# killed is set to True once ctrl + C is pressed
killed = False

# Note: When started is True, and stopped is True, and buffer is empty,
# we will exit the program since all audio samples have been played.

sample_rate = None

event = threading.Event()

first_message_time = None


def generated_audio_callback(samples: np.ndarray, progress: float):
    """This function is called whenever max_num_sentences sentences
    have been processed.

    Note that it is passed to C++ and is invoked in C++.

    Args:
      samples:
        A 1-D np.float32 array containing audio samples
    """
    global first_message_time
    if first_message_time is None:
        first_message_time = time.time()

    buffer.put(samples)
    global started

    if started is False:
        logging.info("Start playing ...")
    started = True

    # 1 means to keep generating
    # 0 means to stop generating
    if killed:
        return 0

    return 1


# see https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#sounddevice.OutputStream
def play_audio_callback(
    outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
):
    if killed or (started and buffer.empty() and stopped):
        event.set()

    # outdata is of shape (frames, num_channels)
    if buffer.empty():
        outdata.fill(0)
        return

    n = 0
    while n < frames and not buffer.empty():
        remaining = frames - n
        k = buffer.queue[0].shape[0]

        if remaining <= k:
            outdata[n:, 0] = buffer.queue[0][:remaining]
            buffer.queue[0] = buffer.queue[0][remaining:]
            n = frames
            if buffer.queue[0].shape[0] == 0:
                buffer.get()

            break

        outdata[n : n + k, 0] = buffer.get()
        n += k

    if n < frames:
        outdata[n:, 0] = 0


# Please see
# https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html#device-selection
# for how to select a device
def play_audio():
    if False:
        # This if branch can be safely removed. It is here to show you how to
        # change the default output device in case you need that.
        devices = sd.query_devices()
        print(devices)

        # sd.default.device[1] is the output device, if you want to
        # select a different device, say, 3, as the output device, please
        # use self.default.device[1] = 3

        default_output_device_idx = sd.default.device[1]
        print(
            f'Use default output device: {devices[default_output_device_idx]["name"]}'
        )

    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=sample_rate,
        blocksize=1024,
    ):
        event.wait()

    logging.info("Exiting ...")


def main():
    args = get_args()
    print(args)

    tts_config = sherpa_mnn.OfflineTtsConfig(
        model=sherpa_mnn.OfflineTtsModelConfig(
            vits=sherpa_mnn.OfflineTtsVitsModelConfig(
                model=args.vits_model,
                lexicon=args.vits_lexicon,
                data_dir=args.vits_data_dir,
                dict_dir=args.vits_dict_dir,
                tokens=args.vits_tokens,
            ),
            matcha=sherpa_mnn.OfflineTtsMatchaModelConfig(
                acoustic_model=args.matcha_acoustic_model,
                vocoder=args.matcha_vocoder,
                lexicon=args.matcha_lexicon,
                tokens=args.matcha_tokens,
                data_dir=args.matcha_data_dir,
                dict_dir=args.matcha_dict_dir,
            ),
            kokoro=sherpa_mnn.OfflineTtsKokoroModelConfig(
                model=args.kokoro_model,
                voices=args.kokoro_voices,
                tokens=args.kokoro_tokens,
                data_dir=args.kokoro_data_dir,
                dict_dir=args.kokoro_dict_dir,
                lexicon=args.kokoro_lexicon,
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        ),
        rule_fsts=args.tts_rule_fsts,
        max_num_sentences=1,
    )

    if not tts_config.validate():
        raise ValueError("Please check your config")

    logging.info("Loading model ...")
    tts = sherpa_mnn.OfflineTts(tts_config)
    logging.info("Loading model done.")

    global sample_rate
    sample_rate = tts.sample_rate

    play_back_thread = threading.Thread(target=play_audio)
    play_back_thread.start()

    logging.info("Start generating ...")
    start_time = time.time()
    audio = tts.generate(
        args.text,
        sid=args.sid,
        speed=args.speed,
        callback=generated_audio_callback,
    )
    end_time = time.time()
    logging.info("Finished generating!")
    global stopped
    stopped = True

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        global killed
        killed = True
        play_back_thread.join()
        return

    elapsed_seconds = end_time - start_time
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write(
        args.output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    logging.info(f"The text is '{args.text}'")
    logging.info(
        "Time in seconds to receive the first "
        f"message: {first_message_time-start_time:.3f}"
    )
    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    logging.info(f"***  Saved to {args.output_filename} ***")

    print("\n   >>>>>>>>> You can safely press ctrl + C to stop the play <<<<<<<<<<\n")

    play_back_thread.join()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
        killed = True
        sys.exit(0)
