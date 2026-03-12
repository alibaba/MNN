//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mnncli_server.hpp"
#include "log_utils.hpp"
#include <iostream>

namespace mnncli {

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

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch); // Find the first non-whitespace character
    });
    return std::string(it, str.end()); // Create a substring from the first non-whitespace character
}

std::string ExtractAnthropicText(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_object()) {
        if (content.contains("type") && content["type"].is_string() &&
            content["type"].get<std::string>() == "text" &&
            content.contains("text") && content["text"].is_string()) {
            return content["text"].get<std::string>();
        }
        return "";
    }
    if (content.is_array()) {
        std::string merged;
        bool first = true;
        for (const auto& block : content) {
            auto text = ExtractAnthropicText(block);
            if (text.empty()) {
                continue;
            }
            if (!first) {
                merged += "\n";
            }
            merged += text;
            first = false;
        }
        return merged;
    }
    return "";
}

json BuildOpenAiMessagesFromAnthropicRequest(const json& request_json) {
    json messages = json::array();

    if (request_json.contains("system")) {
        auto system_text = ExtractAnthropicText(request_json["system"]);
        if (!system_text.empty()) {
            messages.push_back({
                {"role", "system"},
                {"content", system_text}
            });
        }
    }

    if (request_json.contains("messages") && request_json["messages"].is_array()) {
        for (const auto& message : request_json["messages"]) {
            if (!message.is_object()) {
                continue;
            }
            auto role = message.value("role", std::string("user"));
            std::string content_text;
            if (message.contains("content")) {
                content_text = ExtractAnthropicText(message["content"]);
            }
            messages.push_back({
                {"role", role},
                {"content", content_text}
            });
        }
    }

    return messages;
}

void SendSseEvent(httplib::DataSink& sink, const std::string& event, const json& data) {
    std::string chunk = "event: " + event + "\n";
    chunk += "data: " + data.dump() + "\n\n";
    sink.os.write(chunk.c_str(), chunk.size());
    sink.os.flush();
}

    const std::string getR1AssistantString(std::string assistant_content) {
    std::size_t pos = assistant_content.find("</think>");
    if (pos != std::string::npos) {
        assistant_content.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

std::string GetR1UserString(std::string user_content, bool last) {
    return "<|User|>" + std::string(user_content) + "<|Assistant|>";
}

    std::vector<PromptItem> ConvertToR1(std::vector<PromptItem> chat_prompts) {
    std::vector<PromptItem> result_prompts = {};
    std::string prompt_result = "";
    result_prompts.emplace_back("system", "<|begin_of_sentence|>You are a helpful assistant.");
    auto iter = chat_prompts.begin();
    for (; iter != chat_prompts.end() - 1; ++iter) {
        if (iter->first == "system") {
            continue;
        } else if (iter->first == "assistant") {
            result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
        } else if (iter->first == "user") {
            result_prompts.emplace_back("user", GetR1UserString(iter->second, false));
        }
    }
    if (iter->first == "user") {
        result_prompts.emplace_back("user", GetR1UserString(iter->second, true));
    } else {
        result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
    }
    return result_prompts;
}

void MnncliServer::Answer(MNN::Transformer::Llm* llm, const json &messages, std::function<void(const std::string&)> on_result) {
  std::vector<PromptItem> prompts{};
  if (messages.is_array()) {
    for (const auto& item_json : messages) {
      PromptItem item;
      if (!FromJson(item_json, item)) {
        LOG_DEBUG("Error converting JSON object to PromptItem.");
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
  std::ostream output_ostream(&stream_buffer);std::lock_guard<std::mutex> lock(llm_mutex_);
  llm->response(this->is_r1_ ? ConvertToR1(prompts) : prompts, &output_ostream, "<eop>");
}

void MnncliServer::AnswerStreaming(MNN::Transformer::Llm* llm,
                     const json& messages,
                     std::function<void(const std::string&, bool end)> on_partial) {
    std::vector<PromptItem> prompts;
    if (messages.is_array()) {
        for (const auto& item_json : messages) {
            PromptItem item;
            if (!FromJson(item_json, item)) {
                LOG_DEBUG("Error converting JSON object to PromptItem.");
                return;
            }
            prompts.push_back(item);
        }
    }
    std::string answer = "";
    Utf8StreamProcessor processor([&on_partial, &answer](const std::string &utf8Char) {
        bool is_eop = (utf8Char.find("<eop>") != std::string::npos);
        if (is_eop) {
            std::string response_result = answer;
            LOG_DEBUG("response result: " + response_result);
            on_partial("", true);
        } else {
            answer += utf8Char;
            on_partial(utf8Char, false);
        }
    });

    // LlmStreamBuffer calls our lambda as new bytes arrive from the LLM
    LlmStreamBuffer stream_buffer([&processor](const char* str, size_t len) {
        processor.processStream(str, len);
    });
    std::ostream output_ostream(&stream_buffer);
    std::lock_guard<std::mutex> lock(llm_mutex_);
    llm->response(this->is_r1_ ? ConvertToR1(prompts) : prompts, &output_ostream, "<eop>");
}



void AllowCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods",  "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers",  "Content-Type, Authorization");
}

void MnncliServer::Start(MNN::Transformer::Llm* llm, bool is_r1, const std::string& host, int port) {
    this->is_r1_ = is_r1;
    // Create a server instance
    httplib::Server server;

    // Define a route for the GET request on "/"
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        res.set_content(html_content, "text/html");
    });
    server.Post("/reset", [&](const httplib::Request &req, httplib::Response &res) {
      LOG_DEBUG("POST /reset");
      AllowCors(res);
      llm->reset();
      res.set_content("{\"status\": \"ok\"}", "application/json");
    });
    
    server.Get("/v1/models", [&](const httplib::Request &req, httplib::Response &res) {
      LOG_DEBUG("GET /v1/models");
      AllowCors(res);
      json models_response = {
        {"object", "list"},
        {"data", json::array({
          {
            {"id", "ModelScope/MNN/Qwen2.5-0.5B-Instruct"},
            {"object", "model"},
            {"created", static_cast<int>(time(nullptr))},
            {"owned_by", "mnn"}
          }
        })}
      };
      res.set_content(models_response.dump(), "application/json");
    });
    server.Options("/v1/models", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    
    server.Options("/chat/completions", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    
    server.Options("/v1/chat/completions", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    // Handler function for chat completions
    auto chatCompletionsHandler = [&](const httplib::Request &req, httplib::Response &res) {
        LOG_DEBUG("POST chat/completions, handled by thread: " + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
      AllowCors(res);
      if (!json::accept(req.body)) {
          json err;
          err["error"] = "Invalid JSON in request body.";
          res.status = 400;
          res.set_content(err.dump(), "application/json");
          return;
      }
      json request_json = json::parse(req.body, nullptr, false);
      json messages = request_json["messages"];
      LOG_DEBUG("received messages:" + messages.dump(0));
      std::string model = request_json.value("model", "undefined-model");
      bool stream = request_json.value("stream", false);
      if (!stream) {
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
          return;
      }
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      res.set_chunked_content_provider(
            "text/event-stream",
            [llm, messages, model, this](size_t /*offset*/, httplib::DataSink &sink) {
                auto sse_callback = [&, this](const std::string &partial_text, bool end) {
                    std::string finish_reason = end ? "stop" : "";
                    json sse_json = {
                        {"id",       "chatcmpl-" + GetCurrentTimeAsString()},
                        {"object",   "chat.completion.chunk"},
                        {"created",  static_cast<int>(std::time(nullptr))},
                        {"model",    model},
                        {"choices",  json::array({
                            {
                                {"delta", {{"content", partial_text}}},
                                {"index", 0},
                                {"finish_reason", finish_reason}
                            }
                        })}
                    };
                    std::string chunk_str = "data: " + sse_json.dump() + "\n\n";
                    sink.os.write(chunk_str.c_str(), chunk_str.size());
                    sink.os.flush();
                };
                AnswerStreaming(llm, messages, sse_callback);
                std::string done_str = "data: [DONE]\n\n";
                sink.os.write(done_str.c_str(), done_str.size());
                sink.os.flush();
                sink.done();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return false;
            }
        );
    };
    
    // Register both endpoints with the same handler
    server.Post("/chat/completions", chatCompletionsHandler);
    server.Post("/v1/chat/completions", chatCompletionsHandler);
    auto anthropicMessagesHandler = [&](const httplib::Request &req, httplib::Response &res) {
      LOG_DEBUG("POST /v1/messages");
      AllowCors(res);
      if (!json::accept(req.body)) {
          json err;
          err["error"] = "Invalid JSON in request body.";
          res.status = 400;
          res.set_content(err.dump(), "application/json");
          return;
      }

      json request_json = json::parse(req.body, nullptr, false);
      json messages = BuildOpenAiMessagesFromAnthropicRequest(request_json);
      std::string model = request_json.value("model", "undefined-model");
      bool stream = request_json.value("stream", false);

      if (!stream) {
          Answer(llm, messages, [&res, model](const std::string& answer) {
              json response_json = {
                  {"id", "msg_" + GetCurrentTimeAsString()},
                  {"type", "message"},
                  {"role", "assistant"},
                  {"content", json::array({
                      {
                          {"type", "text"},
                          {"text", answer}
                      }
                  })},
                  {"model", model},
                  {"stop_reason", "end_turn"},
                  {"stop_sequence", nullptr},
                  {"usage", {
                      {"input_tokens", 0},
                      {"output_tokens", 0}
                  }}
              };
              res.set_content(response_json.dump(), "application/json");
          });
          return;
      }

      const std::string response_id = "msg_" + GetCurrentTimeAsString();
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      res.set_chunked_content_provider(
          "text/event-stream",
          [llm, messages, model, response_id, this](size_t /*offset*/, httplib::DataSink &sink) {
              SendSseEvent(sink, "message_start", {
                  {"type", "message_start"},
                  {"message", {
                      {"id", response_id},
                      {"type", "message"},
                      {"role", "assistant"},
                      {"model", model},
                      {"content", json::array()},
                      {"usage", {
                          {"input_tokens", 0},
                          {"output_tokens", 0}
                      }}
                  }}
              });

              SendSseEvent(sink, "content_block_start", {
                  {"type", "content_block_start"},
                  {"index", 0},
                  {"content_block", {
                      {"type", "text"},
                      {"text", ""}
                  }}
              });

              int output_tokens = 0;
              auto anthropic_sse_callback = [&](const std::string &partial_text, bool end) {
                  if (end) {
                      return;
                  }
                  if (!partial_text.empty()) {
                      output_tokens += 1;
                  }
                  SendSseEvent(sink, "content_block_delta", {
                      {"type", "content_block_delta"},
                      {"index", 0},
                      {"delta", {
                          {"type", "text_delta"},
                          {"text", partial_text}
                      }}
                  });
              };

              AnswerStreaming(llm, messages, anthropic_sse_callback);

              SendSseEvent(sink, "content_block_stop", {
                  {"type", "content_block_stop"},
                  {"index", 0}
              });

              SendSseEvent(sink, "message_delta", {
                  {"type", "message_delta"},
                  {"delta", {
                      {"stop_reason", "end_turn"},
                      {"stop_sequence", nullptr}
                  }},
                  {"usage", {
                      {"input_tokens", 0},
                      {"output_tokens", output_tokens}
                  }}
              });

              SendSseEvent(sink, "message_stop", {
                  {"type", "message_stop"}
              });

              sink.done();
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
              return false;
          }
      );
    };

    server.Post("/v1/messages", anthropicMessagesHandler);
    server.Options("/v1/messages", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    // Start the server on specified host and port
    LOG_DEBUG("✅ Model initialized successfully!");
    LOG_DEBUG("🚀 Server ready at http://" + host + ":" + std::to_string(port));
    LOG_DEBUG("💡 Press Ctrl+C to stop the server");
    if (!server.listen(host.c_str(), port)) {
        LOG_DEBUG("Error: Could not start server on " + host + ":" + std::to_string(port));
    }
}
}
