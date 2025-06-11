// sherpa/cpp_api/websocket/online-websocket-client.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include <chrono>  // NOLINT
#include <fstream>
#include <string>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"
#include "websocketpp/uri.hpp"

using client = websocketpp::client<websocketpp::config::asio_client>;

using message_ptr = client::message_ptr;
using websocketpp::connection_hdl;

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa-mnn using websocket.

Usage:

./bin/sherpa-mnn-online-websocket-client --help

./bin/sherpa-mnn-online-websocket-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  --samples-per-message=8000 \
  --seconds-per-message=0.2 \
  /path/to/foo.wav

It support only wave of with a single channel, 16kHz, 16-bit samples.
)";

class Client {
 public:
  Client(asio::io_context &io,  // NOLINT
         const std::string &ip, int16_t port, const std::vector<float> &samples,
         int32_t samples_per_message, float seconds_per_message)
      : io_(io),
        uri_(/*secure*/ false, ip, port, /*resource*/ "/"),
        samples_(samples),
        samples_per_message_(samples_per_message),
        seconds_per_message_(seconds_per_message) {
    c_.clear_access_channels(websocketpp::log::alevel::all);
    // c_.set_access_channels(websocketpp::log::alevel::connect);
    // c_.set_access_channels(websocketpp::log::alevel::disconnect);

    c_.init_asio(&io_);
    c_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });
    c_.set_close_handler(
        [](connection_hdl /*hdl*/) { SHERPA_ONNX_LOGE("Disconnected"); });
    c_.set_message_handler(
        [this](connection_hdl hdl, message_ptr msg) { OnMessage(hdl, msg); });

    Run();
  }

 private:
  void Run() {
    websocketpp::lib::error_code ec;
    client::connection_ptr con = c_.get_connection(uri_.str(), ec);
    if (ec) {
      SHERPA_ONNX_LOGE("Could not create connection to %s because %s",
                       uri_.str().c_str(), ec.message().c_str());
      exit(EXIT_FAILURE);
    }

    c_.connect(con);
  }

  void OnOpen(connection_hdl hdl) {
    auto start_time = std::chrono::steady_clock::now();
    asio::post(
        io_, [this, hdl, start_time]() { this->SendMessage(hdl, start_time); });
  }

  void OnMessage(connection_hdl hdl, message_ptr msg) {
    const std::string &payload = msg->get_payload();

    if (payload == "Done!") {
      websocketpp::lib::error_code ec;
      c_.close(hdl, websocketpp::close::status::normal, "I'm exiting now", ec);
      if (ec) {
        SHERPA_ONNX_LOGE("Failed to close because %s", ec.message().c_str());
        exit(EXIT_FAILURE);
      }
    } else {
      SHERPA_ONNX_LOGE("%s", payload.c_str());
    }
  }

  void SendMessage(
      connection_hdl hdl,
      std::chrono::time_point<std::chrono::steady_clock> start_time) {
    int32_t num_samples = samples_.size();
    int32_t num_messages = num_samples / samples_per_message_;

    websocketpp::lib::error_code ec;
    auto time = std::chrono::steady_clock::now();
    int elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time)
            .count();

    if (elapsed_time_ms <
        static_cast<int>(seconds_per_message_ * num_sent_messages_ * 1000)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(int(
          seconds_per_message_ * num_sent_messages_ * 1000 - elapsed_time_ms)));
    }

    if (num_sent_messages_ < 1) {
      SHERPA_ONNX_LOGE("Starting to send audio");
    }

    if (num_sent_messages_ < num_messages) {
      c_.send(hdl, samples_.data() + num_sent_messages_ * samples_per_message_,
              samples_per_message_ * sizeof(float),
              websocketpp::frame::opcode::binary, ec);

      if (ec) {
        SHERPA_ONNX_LOGE("Failed to send audio samples because %s",
                         ec.message().c_str());
        exit(EXIT_FAILURE);
      }

      ec.clear();

      ++num_sent_messages_;
    }

    if (num_sent_messages_ == num_messages) {
      int32_t remaining_samples = num_samples % samples_per_message_;
      if (remaining_samples) {
        c_.send(hdl,
                samples_.data() + num_sent_messages_ * samples_per_message_,
                remaining_samples * sizeof(float),
                websocketpp::frame::opcode::binary, ec);

        if (ec) {
          SHERPA_ONNX_LOGE("Failed to send audio samples because %s",
                           ec.message().c_str());
          exit(EXIT_FAILURE);
        }
        ec.clear();
      }

      // To signal that we have send all the messages
      c_.send(hdl, "Done", websocketpp::frame::opcode::text, ec);
      SHERPA_ONNX_LOGE("Sent Done Signal");

      if (ec) {
        SHERPA_ONNX_LOGE("Failed to send audio samples because %s",
                         ec.message().c_str());
        exit(EXIT_FAILURE);
      }
    } else {
      asio::post(io_, [this, hdl, start_time]() {
        this->SendMessage(hdl, start_time);
      });
    }
  }

 private:
  client c_;
  asio::io_context &io_;
  websocketpp::uri uri_;
  std::vector<float> samples_;
  int32_t samples_per_message_ = 8000;  // 0.5 seconds
  float seconds_per_message_ = 0.2;
  int32_t num_sent_messages_ = 0;
};

int32_t main(int32_t argc, char *argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;

  // Sample rate of the input wave. No resampling is made.
  int32_t sample_rate = 16000;
  int32_t samples_per_message = 8000;
  float seconds_per_message = 0.2;

  sherpa_mnn::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the websocket server");
  po.Register("server-port", &server_port, "Port of the websocket server");
  po.Register("sample-rate", &sample_rate,
              "Sample rate of the input wave. Should be the one expected by "
              "the server");

  po.Register("samples-per-message", &samples_per_message,
              "Send this number of samples per message.");

  po.Register("seconds-per-message", &seconds_per_message,
              "We will simulate that each message takes this number of seconds "
              "to send. If you select a very large value, it will take a long "
              "time to send all the samples");

  po.Read(argc, argv);

  if (!websocketpp::uri_helper::ipv4_literal(server_ip.begin(),
                                             server_ip.end())) {
    SHERPA_ONNX_LOGE("Invalid server IP: %s", server_ip.c_str());
    return -1;
  }

  if (server_port <= 0 || server_port > 65535) {
    SHERPA_ONNX_LOGE("Invalid server port: %d", server_port);
    return -1;
  }

  // 0.01 is an arbitrary value. You can change it.
  if (samples_per_message <= 0.01 * sample_rate) {
    SHERPA_ONNX_LOGE("--samples-per-message is too small: %d",
                     samples_per_message);
    return -1;
  }

  // 100 is an arbitrary value. You can change it.
  if (samples_per_message >= sample_rate * 100) {
    SHERPA_ONNX_LOGE("--samples-per-message is too small: %d",
                     samples_per_message);
    return -1;
  }

  if (seconds_per_message < 0) {
    SHERPA_ONNX_LOGE("--seconds-per-message is too small: %.3f",
                     seconds_per_message);
    return -1;
  }

  // 1 is an arbitrary value.
  if (seconds_per_message > 1) {
    SHERPA_ONNX_LOGE(
        "--seconds-per-message is too large: %.3f. You will wait a long time "
        "to "
        "send all the samples",
        seconds_per_message);
    return -1;
  }

  if (po.NumArgs() != 1) {
    po.PrintUsage();
    return -1;
  }

  std::string wave_filename = po.GetArg(1);

  bool is_ok = false;
  int32_t actual_sample_rate = -1;
  std::vector<float> samples =
      sherpa_mnn::ReadWave(wave_filename, &actual_sample_rate, &is_ok);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read '%s'", wave_filename.c_str());
    return -1;
  }

  if (actual_sample_rate != sample_rate) {
    SHERPA_ONNX_LOGE("Expected sample rate: %d, given %d", sample_rate,
                     actual_sample_rate);
    return -1;
  }

  asio::io_context io_conn;  // for network connections
  Client c(io_conn, server_ip, server_port, samples, samples_per_message,
           seconds_per_message);

  io_conn.run();  // will exit when the above connection is closed

  SHERPA_ONNX_LOGE("Done!");
  return 0;
}
