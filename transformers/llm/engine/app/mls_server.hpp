//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once
#include "llm/llm.hpp"
#include "httplib.h"
#include "jsonhpp/json.hpp"
using nlohmann::json;
using PromptItem = std::pair<std::string, std::string>;
namespace mls {
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
class MlsServer {
  public:
    const char* html_content = R"""(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MNN Frontend</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 2rem;
    }
    #chat-container {
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 1rem;
      height: 400px;
      overflow-y: auto;
      margin-bottom: 1rem;
    }
    .message {
      margin: 0.5rem 0;
    }
    .user {
      color: #333;
      font-weight: bold;
    }
    .assistant {
      color: #007bff;
    }
    #user-input {
      width: 80%;
      padding: 0.5rem;
      font-size: 1rem;
    }
    .button {
      padding: 0.5rem 1rem;
      font-size: 1rem;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h1>Chat with MNN</h1>
  <h3>MNN-LLM's server API is OpenAI API compatible. You can use other frameworks like OpenWebUI or LobeChat.</h3>
  <div id="chat-container"></div>

  <input
    type="text"
    id="user-input"
    placeholder="Type your message here..."
    onkeydown="if(event.key==='Enter'){ sendMessage(); }"
  />
  <br />
  <button id="send-btn" class="button" onclick="sendMessage()">Send</button>
  <button id="reset-btn" class="button" onclick="resetChat()">Reset</button>

  <script>
    const OPENAI_API_KEY = "no";  // put your real key or leave "no" if your server doesn't check it
    const OPENAI_MODEL = "unknown";
    let messages = [
      { role: "system", content: "You are a helpful assistant." },
    ];

    async function resetChat() {
      document.getElementById("user-input").value = "";
      document.getElementById("chat-container").innerHTML = "";
      messages = [];

      await fetch("/reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify({ reset: true }),
      });
    }

    async function sendMessage() {
      const userInput = document.getElementById("user-input").value.trim();
      if (!userInput) return;

      // Display user message
      displayMessage(userInput, "user");
      document.getElementById("user-input").value = "";

      messages.push({ role: "user", content: userInput });

      try {
        // We set "stream": true to indicate we want SSE streaming from our server
        const payload = {
          model: OPENAI_MODEL,
          messages: messages,
          max_tokens: 100,
          temperature: 0.7,
          stream: true,
        };

        const response = await fetch("/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
            Accept: "text/event-stream",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Error: ${response.status} - ${response.statusText}`);
        }

        // Prepare to stream the response
        await handleStream(response);

      } catch (error) {
        displayMessage(`Error: ${error.message}`, "assistant");
      }
    }

    async function handleStream(response) {
      // We'll accumulate tokens into this variable
      let assistantMessage = "";

      // Create a DOM element for the assistant's streaming message
      const chatContainer = document.getElementById("chat-container");
      const messageElem = document.createElement("div");
      messageElem.classList.add("message", "assistant");
      messageElem.innerHTML = `<strong class="assistant">Assistant:</strong> <span></span>`;
      chatContainer.appendChild(messageElem);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      const messageTextSpan = messageElem.querySelector("span");

      // Read the response body as a stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");
          for (let line of lines) {
            if (!line || !line.startsWith("data: ")) {
              continue;
            }
            const jsonStr = line.substring("data: ".length).trim();
            if (jsonStr === "[DONE]") {
              messages.push({ role: "assistant", content: assistantMessage });
              return;
            }

            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.choices && parsed.choices.length > 0) {
                const deltaContent = parsed.choices[0].delta.content;
                if (deltaContent) {
                  assistantMessage += deltaContent;
                  // Update the DOM text
                  messageTextSpan.textContent = assistantMessage;
                  chatContainer.scrollTop = chatContainer.scrollHeight;
                }
              }
            } catch (e) {
              console.error("Could not parse SSE line:", e, line);
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    function displayMessage(text, sender) {
      const chatContainer = document.getElementById("chat-container");
      const messageElem = document.createElement("div");
      messageElem.classList.add("message", sender);

      if (sender === "user") {
        messageElem.innerHTML = `<strong class="user">User:</strong> ${text}`;
      } else {
        messageElem.innerHTML = `<strong class="assistant">Assistant:</strong> ${text}`;
      }

      chatContainer.appendChild(messageElem);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  </script>
</body>
</html>
    )""";
    void Start(MNN::Transformer::Llm* llm, bool is_r1);
    bool is_r1_{false};
private:
  void Answer(MNN::Transformer::Llm* llm, const json &messages, std::function<void(const std::string&)> on_result);
  void AnswerStreaming(MNN::Transformer::Llm* llm,
                     const json& messages,
                     std::function<void(const std::string&, bool end)> on_partial);
    std::mutex llm_mutex_;

};
}
