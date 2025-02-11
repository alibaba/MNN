//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mls_server.hpp"
#include <iostream>
#include "httplib.h"
#include "jsonhpp/json.hpp"

using nlohmann::json;
using PromptItem = std::pair<std::string, std::string>;

namespace mls {

std::string GetCurrentTimeAsString() {
  // Get the current time since epoch
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  // Convert to string
  return std::to_string(seconds);
}

bool FromJson(const json& j, PromptItem& item) {
  if (!j.is_object()) {
    return false;
  }
  if (!j.contains("role") || !j["role"].is_string()) {
    return false;
  }
  if (!j.contains("content") || !j["content"].is_string()) {
    return false;
  }

  item.first = j["role"].get<std::string>();   // Role
  item.second = j["content"].get<std::string>(); // Content
  return true;
}

void Answer(MNN::Transformer::Llm* llm, const json &messages, std::function<void(const std::string&)> on_result) {
  std::vector<PromptItem> prompts{};
  if (messages.is_array()) {
    for (const auto& item_json : messages) {
      PromptItem item;
      if (!FromJson(item_json, item)) {
        std::cerr << "Error converting JSON object to PromptItem." << std::endl;
        break;
      }
      prompts.push_back(item);
    }
  }
  std::stringstream response_buffer;
  Utf8StreamProcessor processor([&response_buffer, on_result](const std::string& utf8Char) {
    bool is_eop = utf8Char.find("<eop>") != std::string::npos;
    if (!is_eop) {
        response_buffer << utf8Char;
    } else {
        std::string response_result =  response_buffer.str();
        on_result(response_result);
    }
    }
  );
  LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
    processor.processStream(str, len);
  }};
  std::ostream output_ostream(&stream_buffer);
  llm->response(prompts, &output_ostream, "<eop>");
}

void MlsServer::Start(MNN::Transformer::Llm* llm) {
    // Create a server instance
    httplib::Server server;

    // Define a route for the GET request on "/"
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_content(html_content, "text/html");
    });
    server.Post("/reset", [&](const httplib::Request &req, httplib::Response &res) {
      printf("POST /reset\n");
      llm->reset();
      res.set_content("{\"status\": \"ok\"}", "application/json");
    });
    server.Post("/v1/chat/completions", [&](const httplib::Request &req, httplib::Response &res) {
      printf("POST /v1/chat/completions\n");
      if (!json::accept(req.body)) {
          // Invalid JSON
          json err;
          err["error"] = "Invalid JSON in request body.";
          res.status = 400;
          res.set_content(err.dump(), "application/json");
          return;
      }
      json request_json = json::parse(req.body, nullptr, false);
      json messages = request_json["messages"];
      std::string model = request_json.value("model", "undefined-model");
      Answer(llm, messages, [&res, model](const std::string& answer) {
          json response_json = {
          {"id", "chatcmpl" + GetCurrentTimeAsString()},
          {"object", "chat.completion"},
          {"created",  static_cast<int>(time(nullptr))},
          {"model", model},
          {
            "choices", json::array({
              {
                {"index", 0},
                {
                  "message", {
                    {"role", "assistant"},
                    {"content", answer}
                  }
                },
                {"finish_reason", "stop"}
              }
            })
          },
          {
            "usage", {
              {"prompt_tokens", 10},
              {"completion_tokens", 7},
              {"total_tokens", 17}
            }
          }
        };
        res.set_content(response_json.dump(), "application/json");
      });
    });
    // Start the server on port 8080
    std::cout << "Starting server on http://localhost:9090\n";
    if (!server.listen("0.0.0.0", 9090)) {
        std::cerr << "Error: Could not start server.\n";
    }
}
}