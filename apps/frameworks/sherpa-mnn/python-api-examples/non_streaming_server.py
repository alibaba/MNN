#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.
"""
A server for non-streaming speech recognition. Non-streaming means you send all
the content of the audio at once for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./non_streaming_server.py --help

Please refer to

https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html

for pre-trained models to download.

Usage examples:

(1) Use a non-streaming transducer model

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
rm sherpa-onnx-zipformer-en-2023-06-26.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
  --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
  --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
  --port 6006
  
(2) Use a non-streaming paraformer

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --paraformer ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt

(3) Use a non-streaming CTC model from NeMo

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
tar xvf sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
rm sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt

(4) Use a non-streaming CTC model from WeNet

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zh-wenet-wenetspeech.tar.bz2
tar xvf sherpa-onnx-zh-wenet-wenetspeech.tar.bz2
rm sherpa-onnx-zh-wenet-wenetspeech.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --wenet-ctc ./sherpa-onnx-zh-wenet-wenetspeech/model.onnx \
  --tokens ./sherpa-onnx-zh-wenet-wenetspeech/tokens.txt

(5) Use a Moonshine model

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --moonshine-preprocessor=./sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx \
  --moonshine-encoder=./sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx \
  --moonshine-uncached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx \
  --moonshine-cached-decoder=./sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx \
  --tokens=./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt

(6) Use a Whisper model

cd /path/to/sherpa-onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt

(7) Use a tdnn model of the yesno recipe from icefall

cd /path/to/sherpa-onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-tdnn-yesno.tar.bz2
tar xvf sherpa-onnx-tdnn-yesno.tar.bz2
rm sherpa-onnx-tdnn-yesno.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --sample-rate=8000 \
  --feat-dim=23 \
  --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
  --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt

(8) Use a Non-streaming SenseVoice model

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt

(9) Use a Non-streaming telespeech ctc model

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
tar xvf sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
rm sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2

python3 ./python-api-examples/non_streaming_server.py \
  --telespeech-ctc=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx \
  --tokens=./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt

----

To use a certificate so that you can use https, please use

python3 ./python-api-examples/non_streaming_server.py \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --certificate=/path/to/your/cert.pem

If you don't have a certificate, please run:

    cd ./python-api-examples/web
    ./generate-certificate.py

It will generate 3 files, one of which is the required `cert.pem`.
"""  # noqa

import argparse
import asyncio
import http
import logging
import socket
import ssl
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sherpa_mnn

import websockets

from http_server import HttpServer


def setup_logger(
    log_filename: str,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    log_filename = f"{log_filename}-{date_time}.txt"

    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


def add_transducer_model_args(parser: argparse.ArgumentParser):
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


def add_paraformer_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )


def add_sense_voice_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )


def add_nemo_ctc_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--nemo-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from NeMo CTC",
    )


def add_telespeech_ctc_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--telespeech-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from TeleSpeech CTC",
    )


def add_wenet_ctc_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from WeNet CTC",
    )


def add_tdnn_ctc_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tdnn-model",
        default="",
        type=str,
        help="Path to the model.onnx for the tdnn model of the yesno recipe",
    )


def add_moonshine_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--moonshine-preprocessor",
        default="",
        type=str,
        help="Path to moonshine preprocessor model",
    )

    parser.add_argument(
        "--moonshine-encoder",
        default="",
        type=str,
        help="Path to moonshine encoder model",
    )

    parser.add_argument(
        "--moonshine-uncached-decoder",
        default="",
        type=str,
        help="Path to moonshine uncached decoder model",
    )

    parser.add_argument(
        "--moonshine-cached-decoder",
        default="",
        type=str,
        help="Path to moonshine cached decoder model",
    )


def add_whisper_model_args(parser: argparse.ArgumentParser):
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
        help="""It specifies the spoken language in the input audio file.
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


def add_model_args(parser: argparse.ArgumentParser):
    add_transducer_model_args(parser)
    add_paraformer_model_args(parser)
    add_sense_voice_model_args(parser)
    add_nemo_ctc_model_args(parser)
    add_wenet_ctc_model_args(parser)
    add_telespeech_ctc_model_args(parser)
    add_tdnn_ctc_model_args(parser)
    add_whisper_model_args(parser)
    add_moonshine_model_args(parser)

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads to run the neural network model",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )


def add_feature_config_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the data used to train the model. ",
    )

    parser.add_argument(
        "--feat-dim",
        type=int,
        default=80,
        help="Feature dimension of the model",
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Current supported methods are:
        - greedy_search
        - modified_beam_search  (for transducer models only)
        """,
    )

    add_modified_beam_search_args(parser)


def add_modified_beam_search_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )


def add_hotwords_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
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


def add_blank_penalty_args(parser: argparse.ArgumentParser):
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


def check_args(args):
    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.decoding_method not in (
        "greedy_search",
        "modified_beam_search",
    ):
        raise ValueError(f"Unsupported decoding method {args.decoding_method}")

    if args.decoding_method == "modified_beam_search":
        assert args.num_active_paths > 0, args.num_active_paths
        assert Path(args.encoder).is_file(), args.encoder
        assert Path(args.decoder).is_file(), args.decoder
        assert Path(args.joiner).is_file(), args.joiner

    if args.hotwords_file != "":
        assert args.decoding_method == "modified_beam_search", args.decoding_method
        assert Path(args.hotwords_file).is_file(), args.hotwords_file


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_model_args(parser)
    add_feature_config_args(parser)
    add_decoding_args(parser)
    add_hotwords_args(parser)
    add_blank_penalty_args(parser)

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=3,
        help="""Max batch size for computation. Note if there are not enough
        requests in the queue, it will wait for max_wait_ms time. After that,
        even if there are not enough requests, it still sends the
        available requests in the queue for computation.
        """,
    )

    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=5,
        help="""Max time in millisecond to wait to build batches for inference.
        If there are not enough requests in the feature queue to build a batch
        of max_batch_size, it waits up to this time before fetching available
        requests for computation.
        """,
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="Number of threads for NN computation and decoding.",
    )

    parser.add_argument(
        "--max-message-size",
        type=int,
        default=(1 << 20),
        help="""Max message size in bytes.
        The max size per message cannot exceed this limit.
        """,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=32,
        help="Max number of messages in the queue for each connection.",
    )

    parser.add_argument(
        "--max-active-connections",
        type=int,
        default=200,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    parser.add_argument(
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use ./web/generate-certificate.py
        to generate the certificate `cert.pem`.
        Note ./web/generate-certificate.py will generate three files but you
        only need to pass the generated cert.pem to this option.
        """,
    )

    parser.add_argument(
        "--doc-root",
        type=str,
        default="./python-api-examples/web",
        help="Path to the web root",
    )

    return parser.parse_args()


class NonStreamingServer:
    def __init__(
        self,
        recognizer: sherpa_mnn.OfflineRecognizer,
        max_batch_size: int,
        max_wait_ms: float,
        nn_pool_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        doc_root: str,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          recognizer:
            An instance of the sherpa_mnn.OfflineRecognizer.
          max_batch_size:
            Max batch size for inference.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `max_batch_size`.
          nn_pool_size:
            Number of threads for the thread pool that is used for NN
            computation and decoding.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          doc_root:
            Path to the directory where files like index.html for the HTTP
            server locate.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        self.recognizer = recognizer

        self.certificate = certificate
        self.http_server = HttpServer(doc_root)

        self.nn_pool_size = nn_pool_size
        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.stream_queue = asyncio.Queue()

        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0
        self.sample_rate = int(recognizer.config.feat_config.sampling_rate)

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if "sec-websocket-key" not in (
            request_headers.headers  # For new request_headers
            if hasattr(request_headers, "headers")
            else request_headers  # For old request_headers
        ):
            # This is a normal HTTP request
            if path == "/":
                path = "/index.html"
            if path[-1] == "?":
                path = path[:-1]

            if path == "/streaming_record.html":
                response = r"""
<!doctype html><html><head>
<title>Speech recognition with next-gen Kaldi</title><body>
<h2>Only
<a href="/upload.html">/upload.html</a>
and
<a href="/offline_record.html">/offline_record.html</a>
is available for the non-streaming server.<h2>
<br/>
<br/>
Go back to <a href="/upload.html">/upload.html</a>
or <a href="/offline_record.html">/offline_record.html</a>
</body></head></html>
"""
                found = True
                mime_type = "text/html"
            else:
                found, response, mime_type = self.http_server.process_request(path)
            if isinstance(response, str):
                response = response.encode("utf-8")

            if not found:
                status = http.HTTPStatus.NOT_FOUND
            else:
                status = http.HTTPStatus.OK
            header = {"Content-Type": mime_type}
            return status, header, response

        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response

    async def run(self, port: int):
        logging.info("started")

        tasks = []
        for i in range(self.nn_pool_size):
            tasks.append(asyncio.create_task(self.stream_consumer_task()))

        if self.certificate:
            logging.info(f"Using certificate: {self.certificate}")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logging.info("No certificate provided")

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
            ssl=ssl_context,
        ):
            ip_list = ["localhost"]
            if ssl_context:
                ip_list += ["0.0.0.0", "127.0.0.1"]
                ip_list.append(socket.gethostbyname(socket.gethostname()))

            proto = "http://" if ssl_context is None else "https://"
            s = "Please visit one of the following addresses:\n\n"
            for p in ip_list:
                s += "  " + proto + p + f":{port}" "\n"
            logging.info(s)

            await asyncio.Future()  # run forever

        await asyncio.gather(*tasks)  # not reachable

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Receive a tensor from the client.

        The message from the client is a **bytes** buffer.

        The first message can be either "Done" meaning the client won't send
        anything in the future or it can be a buffer containing 8 bytes.
        The first 4 bytes in little endian specifies the sample
        rate of the audio samples; the second 4 bytes in little endian specifies
        the number of bytes in the audio file, which will be sent by the client
        in the subsequent messages.
        Since there is a limit in the message size posed by the websocket
        protocol, the client may send the audio file in multiple messages if the
        audio file is very large.

        The second and remaining messages contain audio samples.

        Please refer to ./offline-websocket-client-decode-files-paralell.py
        and ./offline-websocket-client-decode-files-sequential.py
        for how the client sends the message.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a containing:
            - 1-D np.float32 array containing the audio samples
            - sample rate of the audio samples
          or return (None, None) indicating the end of utterance.
        """
        header = await socket.recv()
        if header == "Done":
            return None, None

        assert len(header) >= 8, (
            "The first message should contain at least 8 bytes."
            + f"Given {len(header)}"
        )

        sample_rate = int.from_bytes(header[:4], "little", signed=True)
        expected_num_bytes = int.from_bytes(header[4:8], "little", signed=True)

        received = []
        num_received_bytes = 0
        if len(header) > 8:
            received.append(header[8:])
            num_received_bytes += len(header) - 8

        if num_received_bytes < expected_num_bytes:
            async for message in socket:
                received.append(message)
                num_received_bytes += len(message)
                if num_received_bytes >= expected_num_bytes:
                    break

        assert num_received_bytes == expected_num_bytes, (
            num_received_bytes,
            expected_num_bytes,
        )

        samples = b"".join(received)
        array = np.frombuffer(samples, dtype=np.float32)
        return array, sample_rate

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the RNN-T model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass

            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.recognizer.decode_streams,
                stream_list,
            )

            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(
        self,
        stream: sherpa_mnn.OfflineStream,
    ) -> None:
        """Put the stream into the queue and wait it to be processed by the
        consumer task.

        Args:
          stream:
            The stream to be processed. Note: It is changed in-place.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.stream_queue.put((stream, future))
        await future

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and sends
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket)
        except websockets.exceptions.ConnectionClosedError:
            logging.info(f"{socket.remote_address} disconnected")
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            logging.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        decoding results back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        while True:
            stream = self.recognizer.create_stream()
            samples, sample_rate = await self.recv_audio_samples(socket)
            if samples is None:
                break
            # stream.accept_samples() runs in the main thread

            stream.accept_waveform(sample_rate, samples)

            await self.compute_and_decode(stream)
            result = stream.result.text
            logging.info(f"result: {result}")

            if result:
                await socket.send(result)
            else:
                # If result is an empty string, send something to the client.
                # Otherwise, socket.send() is a no-op and the client will
                # wait for a reply indefinitely.
                await socket.send("<EMPTY>")


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_mnn.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.nemo_ctc) == 0, args.nemo_ctc
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.telespeech_ctc) == 0, args.telespeech_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

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
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            max_active_paths=args.max_active_paths,
            hotwords_file=args.hotwords_file,
            hotwords_score=args.hotwords_score,
            blank_penalty=args.blank_penalty,
            provider=args.provider,
        )
    elif args.paraformer:
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.nemo_ctc) == 0, args.nemo_ctc
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.telespeech_ctc) == 0, args.telespeech_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_mnn.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            provider=args.provider,
        )
    elif args.sense_voice:
        assert len(args.nemo_ctc) == 0, args.nemo_ctc
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.telespeech_ctc) == 0, args.telespeech_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.sense_voice)
        recognizer = sherpa_mnn.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
        )
    elif args.nemo_ctc:
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.telespeech_ctc) == 0, args.telespeech_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.nemo_ctc)

        recognizer = sherpa_mnn.OfflineRecognizer.from_nemo_ctc(
            model=args.nemo_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            provider=args.provider,
        )
    elif args.wenet_ctc:
        assert len(args.telespeech_ctc) == 0, args.telespeech_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.wenet_ctc)

        recognizer = sherpa_mnn.OfflineRecognizer.from_wenet_ctc(
            model=args.wenet_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            provider=args.provider,
        )
    elif args.telespeech_ctc:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.telespeech_ctc)

        recognizer = sherpa_mnn.OfflineRecognizer.from_telespeech_ctc(
            model=args.telespeech_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            provider=args.provider,
        )
    elif args.whisper_encoder:
        assert len(args.tdnn_model) == 0, args.tdnn_model
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        recognizer = sherpa_mnn.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
            provider=args.provider,
        )
    elif args.tdnn_model:
        assert_file_exists(args.tdnn_model)
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        recognizer = sherpa_mnn.OfflineRecognizer.from_tdnn_ctc(
            model=args.tdnn_model,
            tokens=args.tokens,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            provider=args.provider,
        )
    elif args.moonshine_preprocessor:
        assert_file_exists(args.moonshine_preprocessor)
        assert_file_exists(args.moonshine_encoder)
        assert_file_exists(args.moonshine_uncached_decoder)
        assert_file_exists(args.moonshine_cached_decoder)

        recognizer = sherpa_mnn.OfflineRecognizer.from_moonshine(
            preprocessor=args.moonshine_preprocessor,
            encoder=args.moonshine_encoder,
            uncached_decoder=args.moonshine_uncached_decoder,
            cached_decoder=args.moonshine_cached_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
        )
    else:
        raise ValueError("Please specify at least one model")

    return recognizer


def main():
    args = get_args()
    logging.info(vars(args))
    check_args(args)

    recognizer = create_recognizer(args)

    port = args.port
    max_wait_ms = args.max_wait_ms
    max_batch_size = args.max_batch_size
    nn_pool_size = args.nn_pool_size
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    certificate = args.certificate
    doc_root = args.doc_root

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    if not Path(doc_root).is_dir():
        raise ValueError(f"Directory {doc_root} does not exist")

    non_streaming_server = NonStreamingServer(
        recognizer=recognizer,
        max_wait_ms=max_wait_ms,
        max_batch_size=max_batch_size,
        nn_pool_size=nn_pool_size,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        certificate=certificate,
        doc_root=doc_root,
    )
    asyncio.run(non_streaming_server.run(port))


if __name__ == "__main__":
    log_filename = "log/log-non-streaming-server"
    setup_logger(log_filename)
    main()
