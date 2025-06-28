// sherpa-mnn/csrc/offline-websocket-server-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WEBSOCKET_SERVER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WEBSOCKET_SERVER_IMPL_H_

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/tee-stream.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

namespace sherpa_mnn {

/** Communication protocol
 *
 * The client sends a byte stream to the server. The first 4 bytes in little
 * endian indicates the sample rate of the audio data that the client will send.
 * The next 4 bytes in little endian indicates the total samples in bytes the
 * client will send. The remaining bytes represent audio samples. Each audio
 * sample is a float occupying 4 bytes and is normalized into the range
 * [-1, 1].
 *
 * The byte stream can be broken into arbitrary number of messages.
 * We require that the first message has to be at least 8 bytes so that
 * we can get `sample_rate` and `expected_byte_size` from the first message.
 */
struct ConnectionData {
  // Sample rate of the audio samples the client
  int32_t sample_rate;

  // Number of expected bytes sent from the client
  int32_t expected_byte_size = 0;

  // Number of bytes received so far
  int32_t cur = 0;

  // It saves the received samples from the client.
  // We will **reinterpret_cast** it to float.
  // We expect that data.size() == expected_byte_size
  std::vector<int8_t> data;

  void Clear() {
    sample_rate = 0;
    expected_byte_size = 0;
    cur = 0;
    data.clear();
  }
};

using ConnectionDataPtr = std::shared_ptr<ConnectionData>;

struct OfflineWebsocketDecoderConfig {
  OfflineRecognizerConfig recognizer_config;

  int32_t max_batch_size = 5;

  float max_utterance_length = 300;  // seconds

  void Register(ParseOptions *po);
  void Validate() const;
};

class OfflineWebsocketServer;

class OfflineWebsocketDecoder {
 public:
  /**
   * @param config Configuration for the decoder.
   * @param server **Borrowed** from outside.
   */
  explicit OfflineWebsocketDecoder(OfflineWebsocketServer *server);

  /** Insert received data to the queue for decoding.
   *
   * @param hdl A handle to the connection. We can use it to send the result
   *            back to the client once it finishes decoding.
   * @param d  The received data
   */
  void Push(connection_hdl hdl, ConnectionDataPtr d);

  /** It is called by one of the work thread.
   */
  void Decode();

  const OfflineWebsocketDecoderConfig &GetConfig() const { return config_; }

 private:
  OfflineWebsocketDecoderConfig config_;

  /** When we have received all the data from the client, we put it into
   * this queue; the worker threads will get items from this queue for
   * decoding.
   *
   * Number of items to take from this queue is determined by
   * `--max-batch-size`. If there are not enough items in the queue, we won't
   * wait and take whatever we have for decoding.
   */
  std::mutex mutex_;
  std::deque<std::pair<connection_hdl, ConnectionDataPtr>> streams_;

  OfflineWebsocketServer *server_;  // Not owned
  OfflineRecognizer recognizer_;
};

struct OfflineWebsocketServerConfig {
  OfflineWebsocketDecoderConfig decoder_config;
  std::string log_file = "./log.txt";

  void Register(ParseOptions *po);
  void Validate() const;
};

class OfflineWebsocketServer {
 public:
  OfflineWebsocketServer(asio::io_context &io_conn,  // NOLINT
                         asio::io_context &io_work,  // NOLINT
                         const OfflineWebsocketServerConfig &config);

  asio::io_context &GetConnectionContext() { return io_conn_; }
  server &GetServer() { return server_; }

  void Run(uint16_t port);

  const OfflineWebsocketServerConfig &GetConfig() const { return config_; }

 private:
  void SetupLog();

  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // When a websocket client is disconnected, it will invoke this method
  void OnClose(connection_hdl hdl);

  // When a message is received from a websocket client, this method will
  // be invoked.
  //
  // The protocol between the client and the server is as follows:
  //
  // (1) The client connects to the server
  // (2) The client starts to send binary byte stream to the server.
  //     The byte stream can be broken into multiple messages or it can
  //     be put into a single message.
  //     The first message has to contain at least 8 bytes. The first
  //     4 bytes in little endian contains a int32_t indicating the
  //     sampling rate. The next 4 bytes in little endian contains a int32_t
  //     indicating total number of bytes of samples the client will send.
  //     We assume each sample is a float containing 4 bytes and has been
  //     normalized to the range [-1, 1].
  // (4) When the server receives all the samples from the client, it will
  //     start to decode them. Once decoded, the server sends a text message
  //     to the client containing the decoded results
  // (5) After receiving the decoded results from the server, if the client has
  //     another audio file to send, it repeats (2), (3), (4)
  // (6) If the client has no more audio files to decode, the client sends a
  //     text message containing "Done" to the server and closes the connection
  // (7) The server receives a text message "Done" and closes the connection
  //
  // Note:
  //  (a) All models in icefall use features extracted from audio samples
  //      normalized to the range [-1, 1]. Please send normalized audio samples
  //      if you use models from icefall.
  //  (b) Only sound files with a single channel is supported
  //  (c) Only audio samples are sent. For instance, if we want to decode
  //      a WAVE file, the RIFF header of the WAVE is not sent.
  void OnMessage(connection_hdl hdl, server::message_ptr msg);

  // Close a websocket connection with given code and reason
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason);

 private:
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
  server server_;

  std::map<connection_hdl, ConnectionDataPtr, std::owner_less<connection_hdl>>
      connections_;
  std::mutex mutex_;

  OfflineWebsocketServerConfig config_;

  std::ofstream log_;
  TeeStream tee_;

  OfflineWebsocketDecoder decoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WEBSOCKET_SERVER_IMPL_H_
