//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mls_server.hpp"
#include <iostream>




namespace mls {

std::string GetCurrentTimeAsString() {
  // Get the current time since epoch
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  // Convert to string
  return std::to_string(seconds);
}

// bool FromJson(const json& j, PromptItem& item) {
//   if (!j.is_object()) {
//     return false;
//   }
//   if (!j.contains("role") || !j["role"].is_string()) {
//     return false;
//   }
//   if (!j.contains("content") || !j["content"].is_string()) {
//     return false;
//   }

//   item.first = j["role"].get<std::string>();   // Role
//   item.second = j["content"].get<std::string>(); // Content
//   return true;
// }

std::string base64_decode(const std::string &ascdata) {
  static const char b64_table[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  static const char reverse_table[128] = {
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
     64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
     64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64
  };

 std::string retval;
 const std::string::const_iterator last = ascdata.end();
 int bits_collected = 0;
 unsigned int accumulator = 0;

 for (std::string::const_iterator i = ascdata.begin(); i != last; ++i) {
    const int c = *i;
    if (::std::isspace(c) || c == '=') {
       // Skip whitespace and padding. Be liberal in what you accept.
       continue;
    }
    if ((c > 127) || (c < 0) || (reverse_table[c] > 63)) {
        MNN_ERROR("Base64 decode auth code failed.\n");
        return "";
    }
    accumulator = (accumulator << 6) | reverse_table[c];
    bits_collected += 6;
    if (bits_collected >= 8) {
       bits_collected -= 8;
       retval += static_cast<char>((accumulator >> bits_collected) & 0xffu);
    }
 }
 return retval;
}

static std::string SaveImageFromDataUrl(const std::string& data_url) {
  auto comma = data_url.find(",");
  std::string b64 = (comma != std::string::npos ? data_url.substr(comma + 1) : data_url);
  auto bytes = base64_decode(b64);
  std::string filename = "image_" + GetCurrentTimeAsString() + ".jpg";
  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ofs.close();
  return filename;
}

bool FromJson(const json& j, PromptItem& item) {
  if (!j.is_object() || !j.contains("role")) return false;
  item.first = j["role"].get<std::string>();
  if (!j.contains("content")) return false;

  // Handle text or array content
  if (j["content"].is_string()) {
      item.second = j["content"].get<std::string>();
  } else if (j["content"].is_array()) {
      std::string combined;
      for (const auto& elem : j["content"]) {
          if (elem.is_object() && elem.value("type", "") == "image_url"
              && elem.contains("image_url")
              && elem["image_url"].contains("url")) {
              std::string data_url = elem["image_url"]["url"].get<std::string>();
              std::string path = SaveImageFromDataUrl(data_url);
              combined += "<img>" + path + "</img>";
          }
          // other types can be handled here
      }
      item.second = combined;
  } else {
      return false;
  }
  return true;
}

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch); // Find the first non-whitespace character
    });
    return std::string(it, str.end()); // Create a substring from the first non-whitespace character
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

void MlsServer::Answer(MNN::Transformer::Llm* llm, const json &messages, std::function<void(const std::string&)> on_result) {
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
  std::ostream output_ostream(&stream_buffer);std::lock_guard<std::mutex> lock(llm_mutex_);
  llm->response(this->is_r1_ ? ConvertToR1(prompts) : prompts, &output_ostream, "<eop>");
}

void MlsServer::AnswerStreaming(MNN::Transformer::Llm* llm,
                     const json& messages,
                     std::function<void(const std::string&, bool end)> on_partial) {
    std::vector<PromptItem> prompts;
    if (messages.is_array()) {
        for (const auto& item_json : messages) {
            PromptItem item;
            if (!FromJson(item_json, item)) {
                std::cerr << "Error converting JSON object to PromptItem." << std::endl;
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
            std::cout<<"response result: "<<response_result<<std::endl;
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

void MlsServer::Start(MNN::Transformer::Llm* llm, bool is_r1) {
    this->is_r1_ = is_r1;
    // Create a server instance
    httplib::Server server;

    // Define a route for the GET request on "/"
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        res.set_content(html_content, "text/html");
    });
    server.Post("/reset", [&](const httplib::Request &req, httplib::Response &res) {
      printf("POST /reset\n");
      AllowCors(res);
      llm->reset();
      res.set_content("{\"status\": \"ok\"}", "application/json");
    });
    server.Options("/chat/completions", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status  = 200;
    });
    server.Post("/chat/completions", [&](const httplib::Request &req, httplib::Response &res) {
        std::cout << "POST /chat/completions, handled by thread: "
                << std::this_thread::get_id() << std::endl;
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
      std::cout<<"received messages:"<<messages.dump(0)<<std::endl;
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
    });
    // Start the server on port
    std::cout << "Starting server on http://localhost:9090\n";
    if (!server.listen("0.0.0.0", 9090)) {
        std::cerr << "Error: Could not start server.\n";
    }
}
}