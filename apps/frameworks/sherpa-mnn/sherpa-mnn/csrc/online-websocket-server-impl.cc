// sherpa-mnn/csrc/online-websocket-server-impl.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-websocket-server-impl.h"

#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/log.h"

namespace sherpa_mnn {

void OnlineWebsocketDecoderConfig::Register(ParseOptions *po) {
  recognizer_config.Register(po);

  po->Register("loop-interval-ms", &loop_interval_ms,
               "It determines how often the decoder loop runs. ");

  po->Register("max-batch-size", &max_batch_size,
               "Max batch size for recognition.");

  po->Register("end-tail-padding", &end_tail_padding,
               "It determines the length of tail_padding at the end of audio.");
}

void OnlineWebsocketDecoderConfig::Validate() const {
  recognizer_config.Validate();
  SHERPA_ONNX_CHECK_GT(loop_interval_ms, 0);
  SHERPA_ONNX_CHECK_GT(max_batch_size, 0);
  SHERPA_ONNX_CHECK_GT(end_tail_padding, 0);
}

void OnlineWebsocketServerConfig::Register(sherpa_mnn::ParseOptions *po) {
  decoder_config.Register(po);

  po->Register("log-file", &log_file,
               "Path to the log file. Logs are "
               "appended to this file");
}

void OnlineWebsocketServerConfig::Validate() const {
  decoder_config.Validate();
}

OnlineWebsocketDecoder::OnlineWebsocketDecoder(OnlineWebsocketServer *server)
    : server_(server),
      config_(server->GetConfig().decoder_config),
      timer_(server->GetWorkContext()) {
  recognizer_ = std::make_unique<OnlineRecognizer>(config_.recognizer_config);
}

std::shared_ptr<Connection> OnlineWebsocketDecoder::GetOrCreateConnection(
    connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = connections_.find(hdl);
  if (it != connections_.end()) {
    return it->second;
  } else {
    // create a new connection
    std::shared_ptr<OnlineStream> s = recognizer_->CreateStream();
    auto c = std::make_shared<Connection>(hdl, s);
    connections_.insert({hdl, c});
    return c;
  }
}

void OnlineWebsocketDecoder::AcceptWaveform(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  float sample_rate = config_.recognizer_config.feat_config.sampling_rate;
  while (!c->samples.empty()) {
    const auto &s = c->samples.front();
    c->s->AcceptWaveform(sample_rate, s.data(), s.size());
    c->samples.pop_front();
  }
}

void OnlineWebsocketDecoder::InputFinished(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);

  float sample_rate = config_.recognizer_config.feat_config.sampling_rate;

  while (!c->samples.empty()) {
    const auto &s = c->samples.front();
    c->s->AcceptWaveform(sample_rate, s.data(), s.size());
    c->samples.pop_front();
  }

  std::vector<float> tail_padding(
      static_cast<int>(config_.end_tail_padding * sample_rate));

  c->s->AcceptWaveform(sample_rate, tail_padding.data(), tail_padding.size());

  c->s->InputFinished();
  c->eof = true;
}

void OnlineWebsocketDecoder::Warmup() const {
  recognizer_->WarmpUpRecognizer(config_.recognizer_config.model_config.warm_up,
                                 config_.max_batch_size);
}

void OnlineWebsocketDecoder::Run() {
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineWebsocketDecoder::ProcessConnections(const asio::error_code &ec) {
  if (ec) {
    SHERPA_ONNX_LOG(FATAL) << "The decoder loop is aborted!";
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<connection_hdl> to_remove;
  for (auto &p : connections_) {
    auto hdl = p.first;
    auto c = p.second;

    // The order of `if` below matters!
    if (!server_->Contains(hdl)) {
      // If the connection is disconnected, we stop processing it
      to_remove.push_back(hdl);
      continue;
    }

    if (active_.count(hdl)) {
      // Another thread is decoding this stream, so skip it
      continue;
    }

    if (!recognizer_->IsReady(c->s.get()) && !c->eof) {
      // this stream has not enough frames to decode, so skip it
      continue;
    }

    if (!recognizer_->IsReady(c->s.get()) && c->eof) {
      // We won't receive samples from the client, so send a Done! to client

      asio::post(server_->GetWorkContext(),
                 [this, hdl = c->hdl]() { server_->Send(hdl, "Done!"); });

      to_remove.push_back(hdl);
      continue;
    }

    // TODO(fangun): If the connection is timed out, we need to also
    // add it to `to_remove`

    // this stream has enough frames and is currently not processed by any
    // threads, so put it into the ready queue
    ready_connections_.push_back(c);

    // In `Decode()`, it will remove hdl from `active_`
    active_.insert(c->hdl);
  }

  for (auto hdl : to_remove) {
    connections_.erase(hdl);
  }

  if (!ready_connections_.empty()) {
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  // Schedule another call
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineWebsocketDecoder::Decode() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (ready_connections_.empty()) {
    // There are no connections that are ready for decoding,
    // so we return directly
    return;
  }

  std::vector<std::shared_ptr<Connection>> c_vec;
  std::vector<OnlineStream *> s_vec;
  while (!ready_connections_.empty() &&
         static_cast<int32_t>(s_vec.size()) < config_.max_batch_size) {
    auto c = ready_connections_.front();
    ready_connections_.pop_front();

    c_vec.push_back(c);
    s_vec.push_back(c->s.get());
  }

  if (!ready_connections_.empty()) {
    // there are too many ready connections but this thread can only handle
    // max_batch_size connections at a time, so we schedule another call
    // to Decode() and let other threads to process the ready connections
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  lock.unlock();
  recognizer_->DecodeStreams(s_vec.data(), s_vec.size());
  lock.lock();

  for (auto c : c_vec) {
    auto result = recognizer_->GetResult(c->s.get());
    if (recognizer_->IsEndpoint(c->s.get())) {
      result.is_final = true;
      recognizer_->Reset(c->s.get());
    }

    if (!recognizer_->IsReady(c->s.get()) && c->eof) {
      result.is_final = true;
    }

    asio::post(server_->GetConnectionContext(),
               [this, hdl = c->hdl, str = result.AsJsonString()]() {
                 server_->Send(hdl, str);
               });
    active_.erase(c->hdl);
  }
}

OnlineWebsocketServer::OnlineWebsocketServer(
    asio::io_context &io_conn, asio::io_context &io_work,
    const OnlineWebsocketServerConfig &config)
    : config_(config),
      io_conn_(io_conn),
      io_work_(io_work),
      log_(config.log_file, std::ios::app),
      tee_(std::cout, log_),
      decoder_(this) {
  SetupLog();

  server_.init_asio(&io_conn_);

  server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

  server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

  server_.set_message_handler(
      [this](connection_hdl hdl, server::message_ptr msg) {
        OnMessage(hdl, msg);
      });
}

void OnlineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
  auto recognizer_config = config_.decoder_config.recognizer_config;
  int32_t warm_up = recognizer_config.model_config.warm_up;
  const std::string &model_type = recognizer_config.model_config.model_type;
  if (0 < warm_up && warm_up < 100) {
    if (model_type == "zipformer2") {
      decoder_.Warmup();
      SHERPA_ONNX_LOGE("Warm up completed : %d times.", warm_up);
    } else {
      SHERPA_ONNX_LOGE("Only Zipformer2 has warmup support for now.");
      SHERPA_ONNX_LOGE("Given: %s", model_type.c_str());
      exit(0);
    }
  } else if (warm_up == 0) {
    SHERPA_ONNX_LOGE("Starting without warmup!");
  } else {
    SHERPA_ONNX_LOGE("Invalid Warm up Value!. Expected 0 < warm_up < 100");
    exit(0);
  }
  decoder_.Run();
}

void OnlineWebsocketServer::SetupLog() {
  server_.clear_access_channels(websocketpp::log::alevel::all);
  // server_.set_access_channels(websocketpp::log::alevel::connect);
  // server_.set_access_channels(websocketpp::log::alevel::disconnect);

  // So that it also prints to std::cout and std::cerr
  server_.get_alog().set_ostream(&tee_);
  server_.get_elog().set_ostream(&tee_);
}

void OnlineWebsocketServer::Send(connection_hdl hdl, const std::string &text) {
  websocketpp::lib::error_code ec;
  if (!Contains(hdl)) {
    return;
  }

  server_.send(hdl, text, websocketpp::frame::opcode::text, ec);
  if (ec) {
    server_.get_alog().write(websocketpp::log::alevel::app, ec.message());
  }
}

void OnlineWebsocketServer::OnOpen(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.insert(hdl);

  std::ostringstream os;
  os << "New connection: "
     << server_.get_con_from_hdl(hdl)->get_remote_endpoint() << ". "
     << "Number of active connections: " << connections_.size() << ".\n";
  SHERPA_ONNX_LOG(INFO) << os.str();
}

void OnlineWebsocketServer::OnClose(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(hdl);

  SHERPA_ONNX_LOG(INFO) << "Number of active connections: "
                        << connections_.size() << "\n";
}

bool OnlineWebsocketServer::Contains(connection_hdl hdl) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.count(hdl);
}

void OnlineWebsocketServer::OnMessage(connection_hdl hdl,
                                      server::message_ptr msg) {
  auto c = decoder_.GetOrCreateConnection(hdl);

  const std::string &payload = msg->get_payload();

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text:
      if (payload == "Done") {
        asio::post(io_work_, [this, c]() { decoder_.InputFinished(c); });
      }
      break;
    case websocketpp::frame::opcode::binary: {
      auto p = reinterpret_cast<const float *>(payload.data());
      int32_t num_samples = payload.size() / sizeof(float);
      std::vector<float> samples(p, p + num_samples);

      {
        std::lock_guard<std::mutex> lock(c->mutex);
        c->samples.push_back(std::move(samples));
      }

      asio::post(io_work_, [this, c]() { decoder_.AcceptWaveform(c); });
      break;
    }
    default:
      break;
  }
}

void OnlineWebsocketServer::Close(connection_hdl hdl,
                                  websocketpp::close::status::value code,
                                  const std::string &reason) {
  auto con = server_.get_con_from_hdl(hdl);

  std::ostringstream os;
  os << "Closing " << con->get_remote_endpoint() << " with reason: " << reason
     << "\n";

  websocketpp::lib::error_code ec;
  server_.close(hdl, code, reason, ec);
  if (ec) {
    os << "Failed to close" << con->get_remote_endpoint() << ". "
       << ec.message() << "\n";
  }
  server_.get_alog().write(websocketpp::log::alevel::app, os.str());
}

}  // namespace sherpa_mnn
