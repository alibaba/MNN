// sherpa-mnn/csrc/offline-websocket-server.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "asio.hpp"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-websocket-server-impl.h"
#include "sherpa-mnn/csrc/parse-options.h"

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa-mnn using websocket.

Usage:

./bin/sherpa-mnn-offline-websocket-server --help

(1) For transducer models

./bin/sherpa-mnn-offline-websocket-server \
  --port=6006 \
  --num-work-threads=5 \
  --tokens=/path/to/tokens.txt \
  --encoder=/path/to/encoder.onnx \
  --decoder=/path/to/decoder.onnx \
  --joiner=/path/to/joiner.onnx \
  --log-file=./log.txt \
  --max-batch-size=5

(2) For Paraformer

./bin/sherpa-mnn-offline-websocket-server \
  --port=6006 \
  --num-work-threads=5 \
  --tokens=/path/to/tokens.txt \
  --paraformer=/path/to/model.onnx \
  --log-file=./log.txt \
  --max-batch-size=5

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)";

int32_t main(int32_t argc, char *argv[]) {
  sherpa_mnn::ParseOptions po(kUsageMessage);

  sherpa_mnn::OfflineWebsocketServerConfig config;

  // the server will listen on this port
  int32_t port = 6006;

  // size of the thread pool for handling network connections
  int32_t num_io_threads = 1;

  // size of the thread pool for neural network computation and decoding
  int32_t num_work_threads = 3;

  po.Register("num-io-threads", &num_io_threads,
              "Thread pool size for network connections.");

  po.Register("num-work-threads", &num_work_threads,
              "Thread pool size for for neural network "
              "computation and decoding.");

  po.Register("port", &port, "The port on which the server will listen.");

  config.Register(&po);
  po.DisableOption("sample-rate");

  if (argc == 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  po.Read(argc, argv);

  if (po.NumArgs() != 0) {
    SHERPA_ONNX_LOGE("Unrecognized positional arguments!");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  config.Validate();

  asio::io_context io_conn;  // for network connections
  asio::io_context io_work;  // for neural network and decoding

  sherpa_mnn::OfflineWebsocketServer server(io_conn, io_work, config);
  server.Run(port);

  SHERPA_ONNX_LOGE("Started!");
  SHERPA_ONNX_LOGE("Listening on: %d", port);
  SHERPA_ONNX_LOGE("Number of work threads: %d", num_work_threads);

  // give some work to do for the io_work pool
  auto work_guard = asio::make_work_guard(io_work);

  std::vector<std::thread> io_threads;

  // decrement since the main thread is also used for network communications
  for (int32_t i = 0; i < num_io_threads - 1; ++i) {
    io_threads.emplace_back([&io_conn]() { io_conn.run(); });
  }

  std::vector<std::thread> work_threads;
  for (int32_t i = 0; i < num_work_threads; ++i) {
    work_threads.emplace_back([&io_work]() { io_work.run(); });
  }

  io_conn.run();

  for (auto &t : io_threads) {
    t.join();
  }

  for (auto &t : work_threads) {
    t.join();
  }

  return 0;
}
