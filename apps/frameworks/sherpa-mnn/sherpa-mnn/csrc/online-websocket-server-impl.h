// sherpa-mnn/csrc/online-websocket-server-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_WEBSOCKET_SERVER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_WEBSOCKET_SERVER_IMPL_H_

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "asio.hpp"
#include "sherpa-mnn/csrc/online-recognizer.h"
#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/tee-stream.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"
using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

namespace sherpa_mnn {

struct Connection {
  // handle to the connection. We can use it to send messages to the client
  connection_hdl hdl;
  std::shared_ptr<OnlineStream> s;

  // set it to true when InputFinished() is called
  bool eof = false;

  // The last time we received a message from the client
  // TODO(fangjun): Use it to disconnect from a client if it is inactive
  // for a specified time.
  std::chrono::steady_clock::time_point last_active;

  std::mutex mutex;  // protect samples

  // Audio samples received from the client.
  //
  // The I/O threads receive audio samples into this queue
  // and invoke work threads to compute features
  std::deque<std::vector<float>> samples;

  Connection() = default;
  Connection(connection_hdl hdl, std::shared_ptr<OnlineStream> s)
      : hdl(hdl), s(s), last_active(std::chrono::steady_clock::now()) {}
};

struct OnlineWebsocketDecoderConfig {
  OnlineRecognizerConfig recognizer_config;

  // It determines how often the decoder loop runs.
  int32_t loop_interval_ms = 10;

  int32_t max_batch_size = 5;

  float end_tail_padding = 0.8;

  void Register(ParseOptions *po);
  void Validate() const;
};

class OnlineWebsocketServer;

class OnlineWebsocketDecoder {
 public:
  /**
   * @param server  Not owned.
   */
  explicit OnlineWebsocketDecoder(OnlineWebsocketServer *server);

  std::shared_ptr<Connection> GetOrCreateConnection(connection_hdl hdl);

  // Compute features for a stream given audio samples
  void AcceptWaveform(std::shared_ptr<Connection> c);

  // signal that there will be no more audio samples for a stream
  void InputFinished(std::shared_ptr<Connection> c);

  void Warmup() const;

  void Run();

 private:
  void ProcessConnections(const asio::error_code &ec);

  /** It is called by one of the worker thread.
   */
  void Decode();

 private:
  OnlineWebsocketServer *server_;  // not owned
  std::unique_ptr<OnlineRecognizer> recognizer_;
  OnlineWebsocketDecoderConfig config_;
  asio::steady_timer timer_;

  // It protects `connections_`, `ready_connections_`, and `active_`
  std::mutex mutex_;

  std::map<connection_hdl, std::shared_ptr<Connection>,
           std::owner_less<connection_hdl>>
      connections_;

  // Whenever a connection has enough feature frames for decoding, we put
  // it in this queue
  std::deque<std::shared_ptr<Connection>> ready_connections_;

  // If we are decoding a stream, we put it in the active_ set so that
  // only one thread can decode a stream at a time.
  std::set<connection_hdl, std::owner_less<connection_hdl>> active_;
};

struct OnlineWebsocketServerConfig {
  OnlineWebsocketDecoderConfig decoder_config;

  std::string log_file = "./log.txt";

  void Register(sherpa_mnn::ParseOptions *po);
  void Validate() const;
};

class OnlineWebsocketServer {
 public:
  explicit OnlineWebsocketServer(asio::io_context &io_conn,  // NOLINT
                                 asio::io_context &io_work,  // NOLINT
                                 const OnlineWebsocketServerConfig &config);

  void Run(uint16_t port);

  const OnlineWebsocketServerConfig &GetConfig() const { return config_; }
  asio::io_context &GetConnectionContext() { return io_conn_; }
  asio::io_context &GetWorkContext() { return io_work_; }
  server &GetServer() { return server_; }

  void Send(connection_hdl hdl, const std::string &text);

  bool Contains(connection_hdl hdl) const;

 private:
  void SetupLog();

  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // When a websocket client is disconnected, it will invoke this method
  void OnClose(connection_hdl hdl);

  void OnMessage(connection_hdl hdl, server::message_ptr msg);

  // Close a websocket connection with given code and reason
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason);

 private:
  OnlineWebsocketServerConfig config_;
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
  server server_;

  std::ofstream log_;
  sherpa_mnn::TeeStream tee_;

  OnlineWebsocketDecoder decoder_;

  mutable std::mutex mutex_;

  std::set<connection_hdl, std::owner_less<connection_hdl>> connections_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_WEBSOCKET_SERVER_IMPL_H_
