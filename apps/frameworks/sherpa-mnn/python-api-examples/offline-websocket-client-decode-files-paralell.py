#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
A websocket client for sherpa-onnx-offline-websocket-server

This file shows how to transcribe multiple
files in parallel. We create a separate connection for transcribing each file.

Usage:
    ./offline-websocket-client-decode-files-parallel.py \
      --server-addr localhost \
      --server-port 6006 \
      /path/to/foo.wav \
      /path/to/bar.wav \
      /path/to/16kHz.wav \
      /path/to/8kHz.wav

(Note: You have to first start the server before starting the client)

You can find the server at
https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-websocket-server.cc

Note: The server is implemented in C++.
"""

import argparse
import asyncio
import logging
import wave
from typing import Tuple

try:
    import websockets
except ImportError:
    print("please run:")
    print("")
    print("  pip install websockets")
    print("")
    print("before you run this script")
    print("")

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=6006,
        help="Port of the server",
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


async def run(
    server_addr: str,
    server_port: int,
    wave_filename: str,
):
    async with websockets.connect(
        f"ws://{server_addr}:{server_port}"
    ) as websocket:  # noqa
        logging.info(f"Sending {wave_filename}")
        samples, sample_rate = read_wave(wave_filename)
        assert isinstance(sample_rate, int)
        assert samples.dtype == np.float32, samples.dtype
        assert samples.ndim == 1, samples.dim
        buf = sample_rate.to_bytes(4, byteorder="little")  # 4 bytes
        buf += (samples.size * 4).to_bytes(4, byteorder="little")
        buf += samples.tobytes()

        payload_len = 10240
        while len(buf) > payload_len:
            await websocket.send(buf[:payload_len])
            buf = buf[payload_len:]

        if buf:
            await websocket.send(buf)

        decoding_results = await websocket.recv()
        logging.info(f"{wave_filename}\n{decoding_results}")

        # to signal that the client has sent all the data
        await websocket.send("Done")


async def main():
    args = get_args()
    logging.info(vars(args))

    server_addr = args.server_addr
    server_port = args.server_port
    sound_files = args.sound_files

    all_tasks = []
    for wave_filename in sound_files:
        task = asyncio.create_task(
            run(
                server_addr=server_addr,
                server_port=server_port,
                wave_filename=wave_filename,
            )
        )
        all_tasks.append(task)

    await asyncio.gather(*all_tasks)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    asyncio.run(main())
