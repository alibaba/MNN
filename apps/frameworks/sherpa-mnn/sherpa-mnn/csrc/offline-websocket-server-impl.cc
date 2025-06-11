// sherpa-mnn/csrc/offline-websocket-server-impl.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-websocket-server-impl.h"

#include <algorithm>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OfflineWebsocketDecoderConfig::Register(ParseOptions *po) {
  recognizer_config.Register(po);

  po->Register("max-batch-size", &max_batch_size,
               "Max batch size for decoding.");

  po->Register(
      "max-utterance-length", &max_utterance_length,
      "Max utterance length in seconds. If we receive an utterance "
      "longer than this value, we will reject the connection. "
      "If you have enough memory, you can select a large value for it.");
}

void OfflineWebsocketDecoderConfig::Validate() const {
  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Error in recongizer config");
    exit(-1);
  }

  if (max_batch_size <= 0) {
    SHERPA_ONNX_LOGE("Expect --max-batch-size > 0. Given: %d", max_batch_size);
    exit(-1);
  }

  if (max_utterance_length <= 0) {
    SHERPA_ONNX_LOGE("Expect --max-utterance-length > 0. Given: %f",
                     max_utterance_length);
    exit(-1);
  }
}

OfflineWebsocketDecoder::OfflineWebsocketDecoder(OfflineWebsocketServer *server)
    : config_(server->GetConfig().decoder_config),
      server_(server),
      recognizer_(config_.recognizer_config) {}

void OfflineWebsocketDecoder::Push(connection_hdl hdl, ConnectionDataPtr d) {
  std::lock_guard<std::mutex> lock(mutex_);
  streams_.push_back({hdl, d});
}

void OfflineWebsocketDecoder::Decode() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (streams_.empty()) {
    return;
  }

  int32_t size =
      std::min(static_cast<int32_t>(streams_.size()), config_.max_batch_size);
  SHERPA_ONNX_LOGE("size: %d", size);

  // We first lock the mutex for streams_, take items from it, and then
  // unlock the mutex; in doing so we don't need to lock the mutex to
  // access hdl and connection_data later.
  std::vector<connection_hdl> handles(size);

  // Store connection_data here to prevent the data from being freed
  // while we are still using it.
  std::vector<ConnectionDataPtr> connection_data(size);

  std::vector<const float *> samples(size);
  std::vector<int32_t> samples_length(size);
  std::vector<std::unique_ptr<OfflineStream>> ss(size);
  std::vector<OfflineStream *> p_ss(size);

  for (int32_t i = 0; i != size; ++i) {
    auto &p = streams_.front();
    handles[i] = p.first;
    connection_data[i] = p.second;
    streams_.pop_front();

    auto sample_rate = connection_data[i]->sample_rate;
    auto samples =
        reinterpret_cast<const float *>(&connection_data[i]->data[0]);
    auto num_samples = connection_data[i]->expected_byte_size / sizeof(float);
    auto s = recognizer_.CreateStream();
    s->AcceptWaveform(sample_rate, samples, num_samples);

    ss[i] = std::move(s);
    p_ss[i] = ss[i].get();
  }

  lock.unlock();

  // Note: DecodeStreams is thread-safe
  recognizer_.DecodeStreams(p_ss.data(), size);

  for (int32_t i = 0; i != size; ++i) {
    connection_hdl hdl = handles[i];
    asio::post(server_->GetConnectionContext(),
               [this, hdl, result = ss[i]->GetResult()]() {
                 websocketpp::lib::error_code ec;
                 server_->GetServer().send(hdl, result.AsJsonString(),
                                           websocketpp::frame::opcode::text,
                                           ec);
                 if (ec) {
                   server_->GetServer().get_alog().write(
                       websocketpp::log::alevel::app, ec.message());
                 }
               });
  }
}

void OfflineWebsocketServerConfig::Register(ParseOptions *po) {
  decoder_config.Register(po);
  po->Register("log-file", &log_file,
               "Path to the log file. Logs are "
               "appended to this file");
}

void OfflineWebsocketServerConfig::Validate() const {
  decoder_config.Validate();
}

OfflineWebsocketServer::OfflineWebsocketServer(
    asio::io_context &io_conn,  // NOLINT
    asio::io_context &io_work,  // NOLINT
    const OfflineWebsocketServerConfig &config)
    : io_conn_(io_conn),
      io_work_(io_work),
      config_(config),
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

void OfflineWebsocketServer::SetupLog() {
  server_.clear_access_channels(websocketpp::log::alevel::all);
  server_.set_access_channels(websocketpp::log::alevel::connect);
  server_.set_access_channels(websocketpp::log::alevel::disconnect);

  // So that it also prints to std::cout and std::cerr
  server_.get_alog().set_ostream(&tee_);
  server_.get_elog().set_ostream(&tee_);
}

void OfflineWebsocketServer::OnOpen(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.emplace(hdl, std::make_shared<ConnectionData>());

  SHERPA_ONNX_LOGE("Number of active connections: %d",
                   static_cast<int32_t>(connections_.size()));
}

void OfflineWebsocketServer::OnClose(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(hdl);

  SHERPA_ONNX_LOGE("Number of active connections: %d",
                   static_cast<int32_t>(connections_.size()));
}

void OfflineWebsocketServer::OnMessage(connection_hdl hdl,
                                       server::message_ptr msg) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto connection_data = connections_.find(hdl)->second;
  lock.unlock();
  const std::string &payload = msg->get_payload();

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text:
      if (payload == "Done") {
        // The client will not send any more data. We can close the
        // connection now.
        Close(hdl, websocketpp::close::status::normal, "Done");
      } else {
        Close(hdl, websocketpp::close::status::normal,
              std::string("Invalid payload: ") + payload);
      }
      break;

    case websocketpp::frame::opcode::binary: {
      auto p = reinterpret_cast<const int8_t *>(payload.data());

      if (connection_data->expected_byte_size == 0) {
        if (payload.size() < 8) {
          Close(hdl, websocketpp::close::status::normal,
                "Payload is too short");
          break;
        }

        connection_data->sample_rate = *reinterpret_cast<const int32_t *>(p);

        connection_data->expected_byte_size =
            *reinterpret_cast<const int32_t *>(p + 4);

        int32_t max_byte_size_ = decoder_.GetConfig().max_utterance_length *
                                 connection_data->sample_rate * sizeof(float);
        if (connection_data->expected_byte_size > max_byte_size_) {
          float num_samples =
              connection_data->expected_byte_size / sizeof(float);

          float duration = num_samples / connection_data->sample_rate;

          std::ostringstream os;
          os << "Max utterance length is configured to "
             << decoder_.GetConfig().max_utterance_length
             << " seconds, received length is " << duration << " seconds. "
             << "Payload is too large!";
          Close(hdl, websocketpp::close::status::message_too_big, os.str());
          break;
        }

        connection_data->data.resize(connection_data->expected_byte_size);
        std::copy(payload.begin() + 8, payload.end(),
                  connection_data->data.data());
        connection_data->cur = payload.size() - 8;
      } else {
        std::copy(payload.begin(), payload.end(),
                  connection_data->data.data() + connection_data->cur);
        connection_data->cur += payload.size();
      }

      if (connection_data->expected_byte_size == connection_data->cur) {
        auto d = std::make_shared<ConnectionData>(std::move(*connection_data));
        // Clear it so that we can handle the next audio file from the client.
        // The client can send multiple audio files for recognition without
        // the need to create another connection.
        connection_data->sample_rate = 0;
        connection_data->expected_byte_size = 0;
        connection_data->cur = 0;

        decoder_.Push(hdl, d);

        connection_data->Clear();

        asio::post(io_work_, [this]() { decoder_.Decode(); });
      }
      break;
    }

    default:
      // Unexpected message, ignore it
      break;
  }
}

void OfflineWebsocketServer::Close(connection_hdl hdl,
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

void OfflineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
}

}  // namespace sherpa_mnn
