#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
A websocket client for sherpa-onnx-online-websocket-server

Usage:
    ./online-websocket-client-microphone.py \
      --server-addr localhost \
      --server-port 6006

(Note: You have to first start the server before starting the client)

You can find the C++ server at
https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-websocket-server.cc
or use the python server ./python-api-examples/streaming_server.py

There is also a C++ version of the client. Please see
https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-websocket-client.cc
"""

import argparse
import asyncio
import sys

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

try:
    import websockets
except ImportError:
    print("please run:")
    print("")
    print("  pip install websockets")
    print("")
    print("before you run this script")
    print("")
    sys.exit(-1)


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

    return parser.parse_args()


async def inputstream_generator(channels=1):
    """Generator that yields blocks of input data as NumPy arrays.

    See https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#creating-an-asyncio-generator-for-audio-blocks
    """
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    print()
    print("Started! Please speak")

    stream = sd.InputStream(
        callback=callback,
        channels=channels,
        dtype="float32",
        samplerate=16000,
        blocksize=int(0.05 * 16000),  # 0.05 seconds
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def receive_results(socket: websockets.WebSocketServerProtocol):
    last_message = ""
    async for message in socket:
        if message != "Done!":
            if last_message != message:
                last_message = message

                if last_message:
                    print(last_message)
        else:
            return last_message


async def run(
    server_addr: str,
    server_port: int,
):
    async with websockets.connect(
        f"ws://{server_addr}:{server_port}"
    ) as websocket:  # noqa
        receive_task = asyncio.create_task(receive_results(websocket))
        print("Started! Please Speak")

        async for indata, status in inputstream_generator():
            if status:
                print(status)
            indata = indata.reshape(-1)
            indata = np.ascontiguousarray(indata)
            await websocket.send(indata.tobytes())

        decoding_results = await receive_task
        print(f"\nFinal result is:\n{decoding_results}")


async def main():
    args = get_args()
    print(vars(args))

    server_addr = args.server_addr
    server_port = args.server_port

    await run(
        server_addr=server_addr,
        server_port=server_port,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
