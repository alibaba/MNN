//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#include "httplib.h"
#include "json.hpp"
using nlohmann::json;
using PromptItem = std::pair<std::string, std::string>;
namespace mnncli {
class LlmStreamBuffer : public std::streambuf {
public:
  using CallBack = std::function<void(const char* str, size_t len)>;
  explicit LlmStreamBuffer(CallBack callback) : callback_(std::move(callback)) {}
  ~LlmStreamBuffer() override = default;
protected:
  virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
    if (callback_) {
      callback_(s, n);
    }
    return n;
  }

private:
  CallBack callback_{};
};

class Utf8StreamProcessor {
  public:
    Utf8StreamProcessor(std::function<void(const std::string&)> callback)
            : callback(callback) {}

    void processStream(const char* str, size_t len) {
      utf8Buffer.append(str, len);

      size_t i = 0;
      std::string completeChars;
      while (i < utf8Buffer.size()) {
        int length = utf8CharLength(static_cast<unsigned char>(utf8Buffer[i]));
        if (length == 0 || i + length > utf8Buffer.size()) {
          break;
        }
        completeChars.append(utf8Buffer, i, length);
        i += length;
      }
      utf8Buffer = utf8Buffer.substr(i);
      if (!completeChars.empty()) {
        callback(completeChars);
      }
    }
    int utf8CharLength(unsigned char byte) {
      if ((byte & 0x80) == 0) return 1;     
      if ((byte & 0xE0) == 0xC0) return 2;
      if ((byte & 0xF0) == 0xE0) return 3;
      if ((byte & 0xF8) == 0xF0) return 4;
      return 0;
    }
  private:
    std::string utf8Buffer;
    std::function<void(const std::string&)> callback;
  };
class MnncliServer {
  public:
      bool Start(MNN::Transformer::Llm* llm, const std::string& model, const std::string& host = "127.0.0.1",
                 int port = 8000, const std::string& api_key = "", MNN::Transformer::Embedding* embedding = nullptr,
                 const std::string& embedding_model = "", const std::string& config_path = "", bool verbose = false);

  private:
      std::mutex llm_mutex_;
};
}
