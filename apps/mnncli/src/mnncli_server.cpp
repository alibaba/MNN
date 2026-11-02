//
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mnncli_server.hpp"
#include "mnncli_agent.hpp"
#include "mnncli_web_ui.hpp"
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <regex>
#include <cv/imgcodecs.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <openssl/evp.h>

namespace mnncli {
namespace {

using MNN::Transformer::Llm;
using MNN::Transformer::LlmStatus;

struct Request {
    std::vector<PromptItem> messages;
    json sampler = json::object();
    json tools = json::array();
    std::map<std::string, MNN::Transformer::PromptImagePart> images;
    bool stream = false;
    bool include_usage = false;
    int max_tokens = -1;
};

struct Result {
    std::string text;
    json tool_calls = json::array();
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int decode_tokens = 0;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    int64_t vision_us = 0;
    LlmStatus status = LlmStatus::INTERNAL_ERROR;
};

std::string responseId(const char* prefix) {
    static std::atomic<unsigned long long> next{0};
    return std::string(prefix) + std::to_string(std::time(nullptr)) + "-" + std::to_string(next++);
}

void allowCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
}

void sendError(httplib::Response& res, int status, const std::string& message, bool anthropic) {
    res.status = status;
    json error = {{"type", status == 401   ? "authentication_error"
                           : status == 404 ? "not_found_error"
                           : status >= 500 ? "api_error"
                                           : "invalid_request_error"},
                  {"message", message}};
    json body = {{"error", error}};
    if (anthropic)
        body["type"] = "error";
    res.set_content(body.dump(), "application/json");
}

void sendEvent(httplib::DataSink& sink, const std::string& event, const json& data) {
    const std::string chunk = (event.empty() ? "" : "event: " + event + "\n") + "data: " + data.dump() + "\n\n";
    sink.write(chunk.data(), chunk.size());
}

bool textContent(const json& content, std::string& text) {
    if (content.is_string()) {
        text = content.get<std::string>();
        return true;
    }
    if (!content.is_array() || content.empty())
        return false;
    for (const auto& block : content) {
        if (!block.is_object() || !block.contains("type") || !block["type"].is_string() || block["type"] != "text" ||
            !block.contains("text") || !block["text"].is_string())
            return false;
        text += block["text"].get<std::string>();
    }
    return true;
}

bool imageContent(const std::string& encoded, Request& request, std::string& text, std::string& error) {
    const auto comma = encoded.find(',');
    if (comma == std::string::npos ||
        (encoded.substr(0, comma) != "data:image/png;base64" && encoded.substr(0, comma) != "data:image/jpeg;base64")) {
        error = "Images must be PNG or JPEG base64 data URLs";
        return false;
    }
    const auto data = encoded.substr(comma + 1);
    if (data.empty() || data.size() > 12 * 1024 * 1024 || data.size() % 4 != 0 || request.images.size() >= 8) {
        error = "Invalid image size or too many images";
        return false;
    }
    size_t padding = data.back() == '=' ? 1 : 0;
    if (data.size() > 1 && data[data.size() - 2] == '=')
        ++padding;
    for (size_t i = 0; i < data.size() - padding; ++i) {
        unsigned char c = data[i];
        if (!std::isalnum(c) && c != '+' && c != '/') {
            error = "Invalid base64 image";
            return false;
        }
    }
    std::vector<uint8_t> bytes(data.size() / 4 * 3);
    int size = EVP_DecodeBlock(bytes.data(), reinterpret_cast<const unsigned char*>(data.data()),
                               static_cast<int>(data.size()));
    if (size <= static_cast<int>(padding)) {
        error = "Invalid base64 image";
        return false;
    }
    bytes.resize(size - padding);
    auto image = MNN::CV::imdecode(bytes, MNN::CV::IMREAD_COLOR, 16 * 1024 * 1024);
    if (!image.get() || !image->getInfo() || image->getInfo()->dim.size() != 3) {
        error = "Could not decode image";
        return false;
    }
    const auto& dims = image->getInfo()->dim;
    const auto key = "mnncli-image-" + std::to_string(request.images.size());
    request.images.emplace(key, MNN::Transformer::PromptImagePart{image, dims[1], dims[0]});
    text += "<img>" + key + "</img>";
    return true;
}

bool messageContent(const json& content, bool anthropic, bool vision, Request& request, std::string& text,
                    std::string& error) {
    if (content.is_string())
        text = content.get<std::string>();
    else if (content.is_array() && !content.empty()) {
        for (const auto& part : content) {
            if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
                error = "Invalid content block";
                return false;
            }
            const auto type = part["type"].get<std::string>();
            if (type == "text" && part.contains("text") && part["text"].is_string()) {
                const auto value = part["text"].get<std::string>();
                if (value.find("<img>") != std::string::npos || value.find("<audio>") != std::string::npos ||
                    value.find("<video>") != std::string::npos) {
                    error = "Raw media tags are not supported";
                    return false;
                }
                text += value;
            } else if (vision && !anthropic && type == "image_url" && part.contains("image_url") &&
                       part["image_url"].is_object() && part["image_url"].contains("url") &&
                       part["image_url"]["url"].is_string()) {
                if (!imageContent(part["image_url"]["url"].get<std::string>(), request, text, error))
                    return false;
            } else if (vision && anthropic && type == "image" && part.contains("source") &&
                       part["source"].is_object() && part["source"].contains("type") &&
                       part["source"]["type"].is_string() && part["source"]["type"] == "base64" &&
                       part["source"].contains("media_type") && part["source"]["media_type"].is_string() &&
                       part["source"].contains("data") && part["source"]["data"].is_string()) {
                auto url = "data:" + part["source"]["media_type"].get<std::string>() + ";base64," +
                           part["source"]["data"].get<std::string>();
                if (!imageContent(url, request, text, error))
                    return false;
            } else {
                error = "Unsupported content block or model does not support images";
                return false;
            }
        }
    } else {
        error = "Message content must be text or content blocks";
        return false;
    }
    if (content.is_string() && (text.find("<img>") != std::string::npos || text.find("<audio>") != std::string::npos ||
                                text.find("<video>") != std::string::npos)) {
        error = "Raw media tags are not supported";
        return false;
    }
    return true;
}

bool parseRequest(const json& body, const std::string& model, bool anthropic, bool tool_capable, bool vision,
                  Request& request, std::string& error) {
    if (!body.is_object()) {
        error = "Request body must be a JSON object";
        return false;
    }
    if (!body.contains("model") || !body["model"].is_string() || body["model"].get<std::string>() != model) {
        error = "Model not found; this server serves " + model;
        return false;
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
        error = "stream must be a boolean";
        return false;
    }
    request.stream = body.value("stream", false);
    const char* unsupported[] = {
        "stop",       "stop_sequences", "response_format", "seed",         "presence_penalty", "frequency_penalty",
        "logit_bias", "logprobs",       "top_logprobs",    "service_tier", "thinking",         "output_config"};
    for (const auto* field : unsupported) {
        if (body.contains(field) && !body[field].is_null()) {
            error = std::string("Unsupported parameter: ") + field;
            return false;
        }
    }
    if (body.contains("tools") && !body["tools"].is_null() && !body["tools"].empty()) {
        if (!tool_capable || !body["tools"].is_array()) {
            error = "Tools require a supported tool-call chat template";
            return false;
        }
        for (const auto& tool : body["tools"]) {
            if (anthropic) {
                if (!tool.is_object() || !tool.contains("name") || !tool["name"].is_string() ||
                    tool["name"].get<std::string>().empty() || !tool.contains("input_schema") ||
                    !tool["input_schema"].is_object()) {
                    error = "Each Anthropic tool requires a name and input_schema";
                    return false;
                }
                json function = {{"name", tool["name"]}, {"parameters", tool["input_schema"]}};
                if (tool.contains("description"))
                    function["description"] = tool["description"];
                request.tools.push_back({{"type", "function"}, {"function", function}});
            } else if (!tool.is_object() || !tool.contains("type") || tool["type"] != "function" ||
                       !tool.contains("function") || !tool["function"].is_object() ||
                       !tool["function"].contains("name") || !tool["function"]["name"].is_string() ||
                       tool["function"]["name"].get<std::string>().empty()) {
                error = "Each tool must contain a function with a name";
                return false;
            } else
                request.tools.push_back(tool);
        }
    } else if (body.contains("tools") && !body["tools"].is_null() && !body["tools"].is_array()) {
        error = "tools must be an array";
        return false;
    }
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
        auto choice = body["tool_choice"];
        if (anthropic && choice.is_object() && choice.contains("type"))
            choice = choice["type"];
        if (!choice.is_string() || (choice != "auto" && choice != "none")) {
            error = "Only tool_choice auto and none are supported";
            return false;
        }
        if (choice == "none")
            request.tools = json::array();
    }
    for (const auto* field : {"temperature", "top_p"}) {
        if (body.contains(field) && !body[field].is_null()) {
            if (!body[field].is_number()) {
                error = std::string(field) + " must be a number";
                return false;
            }
            double value = body[field].get<double>();
            if (value < 0 || (std::string(field) == "temperature" && (value > 2 || (value > 0 && value < 0.00001))) ||
                (std::string(field) == "top_p" && (value == 0 || value > 1))) {
                error = std::string("Invalid ") + field;
                return false;
            }
            request.sampler[field] = value;
        }
    }
    if (body.contains("top_k") && !body["top_k"].is_null()) {
        if (!body["top_k"].is_number_integer() || body["top_k"] <= 0 || body["top_k"] > INT_MAX) {
            error = "top_k must be a positive integer";
            return false;
        }
        request.sampler["top_k"] = body["top_k"];
    }
    if (!request.sampler.empty()) {
        if (!request.sampler.contains("temperature")) request.sampler["temperature"] = 1.0;
        if (request.sampler.value("temperature", 1.0) == 0.0) {
            request.sampler["sampler_type"] = "greedy";
        } else {
            json steps = json::array();
            if (request.sampler.contains("top_k"))
                steps.push_back("topK");
            if (request.sampler.contains("top_p"))
                steps.push_back("topP");
            steps.push_back("temperature");
            request.sampler["sampler_type"] = "mixed";
            request.sampler["mixed_samplers"] = steps;
        }
    }
    if (body.contains("n") && !body["n"].is_null() && body["n"] != 1) {
        error = "Only n=1 is supported";
        return false;
    }
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
        const auto& options = body["stream_options"];
        if (anthropic || !request.stream || !options.is_object() || options.size() != 1 ||
            !options.contains("include_usage") || !options["include_usage"].is_boolean()) {
            error = "Only stream_options.include_usage is supported for streaming chat";
            return false;
        }
        request.include_usage = options["include_usage"].get<bool>();
    }
    if (body.contains("max_tokens") && body.contains("max_completion_tokens") && !body["max_tokens"].is_null() &&
        !body["max_completion_tokens"].is_null()) {
        error = "Specify only one token limit";
        return false;
    }
    const char* limit = body.contains("max_completion_tokens") && !body["max_completion_tokens"].is_null()
                            ? "max_completion_tokens"
                            : "max_tokens";
    if (body.contains(limit) && !body[limit].is_null()) {
        const auto& value = body[limit];
        if (!value.is_number_integer() || value.is_number_unsigned() && value.get<unsigned long long>() > INT_MAX ||
            !value.is_number_unsigned() && (value.get<long long>() <= 0 || value.get<long long>() > INT_MAX) ||
            value.is_number_unsigned() && value.get<unsigned long long>() == 0) {
            error = "Token limit must be a positive integer";
            return false;
        }
        request.max_tokens = value.get<int>();
    } else if (anthropic) {
        error = "max_tokens is required";
        return false;
    }
    if (anthropic && body.contains("system")) {
        std::string system;
        if (!textContent(body["system"], system) || system.find("<img>") != std::string::npos ||
            system.find("<audio>") != std::string::npos || system.find("<video>") != std::string::npos) {
            error = "Unsupported system content";
            return false;
        }
        request.messages.emplace_back("system", system);
    }
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        error = "messages must be a non-empty array";
        return false;
    }
    for (const auto& message : body["messages"]) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string() ||
            !message.contains("content")) {
            error = "Each message requires a role and text content";
            return false;
        }
        const auto role = message["role"].get<std::string>();
        if (role != "user" && role != "assistant" && (anthropic || (role != "system" && role != "tool"))) {
            error = "Unsupported message role: " + role;
            return false;
        }
        std::string content;
        if (anthropic && message["content"].is_array()) {
            bool has_tools = false;
            for (const auto& block : message["content"]) {
                if (block.is_object() && block.contains("type") && block["type"].is_string() &&
                    (block["type"] == "tool_use" || block["type"] == "tool_result"))
                    has_tools = true;
            }
            if (has_tools) {
                if (!tool_capable) {
                    error = "Tool history requires a supported chat template";
                    return false;
                }
                json calls = json::array();
                bool seen_tool_use = false;
                for (const auto& block : message["content"]) {
                    if (!block.is_object() || !block.contains("type") || !block["type"].is_string()) {
                        error = "Invalid tool block";
                        return false;
                    }
                    if (block["type"] == "text" && block.contains("text") && block["text"].is_string()) {
                        if (role == "assistant" && seen_tool_use) {
                            error = "Text after an assistant tool call is not supported";
                            return false;
                        }
                        content += block["text"].get<std::string>();
                    } else if (role == "assistant" && block["type"] == "tool_use" && block.contains("id") &&
                               block["id"].is_string() && block.contains("name") && block["name"].is_string() &&
                               block.contains("input") && block["input"].is_object()) {
                        seen_tool_use = true;
                        calls.push_back(
                            {{"id", block["id"]},
                             {"type", "function"},
                             {"function", {{"name", block["name"]}, {"arguments", block["input"].dump()}}}});
                    } else if (role == "user" && block["type"] == "tool_result" && block.contains("tool_use_id") &&
                               block["tool_use_id"].is_string() && block.contains("content")) {
                        std::string value;
                        if (!textContent(block["content"], value)) {
                            error = "Only text tool results are supported";
                            return false;
                        }
                        if (!content.empty()) {
                            request.messages.emplace_back("user", content);
                            content.clear();
                        }
                        json tool_message = {
                            {"role", "tool"}, {"tool_call_id", block["tool_use_id"]}, {"content", value}};
                        request.messages.emplace_back("json", tool_message.dump());
                    } else {
                        error = "Unsupported tool block";
                        return false;
                    }
                }
                if (role == "assistant") {
                    json assistant = {{"role", "assistant"}, {"content", content}, {"tool_calls", calls}};
                    for (auto& call : assistant["tool_calls"])
                        call["function"]["arguments"] = json::parse(call["function"]["arguments"].get<std::string>());
                    request.messages.emplace_back("json", assistant.dump());
                } else if (!content.empty())
                    request.messages.emplace_back("user", content);
                continue;
            }
        }
        if (role == "assistant" && message["content"].is_null() && message.contains("tool_calls")) {
            content = "";
        } else if (!messageContent(message["content"], anthropic, vision, request, content, error)) {
            return false;
        }
        if (message.contains("tool_calls") || role == "tool") {
            if (anthropic || !tool_capable ||
                (role == "tool" && (!message.contains("tool_call_id") || !message["tool_call_id"].is_string())) ||
                (message.contains("tool_calls") && (role != "assistant" || !message["tool_calls"].is_array()))) {
                error = "Invalid or unsupported tool message";
                return false;
            }
            if (message.contains("tool_calls")) {
                for (const auto& call : message["tool_calls"]) {
                    if (!call.is_object() || !call.contains("type") || call["type"] != "function" ||
                        !call.contains("id") || !call["id"].is_string() || !call.contains("function") ||
                        !call["function"].is_object() || !call["function"].contains("name") ||
                        !call["function"]["name"].is_string() || !call["function"].contains("arguments") ||
                        !call["function"]["arguments"].is_string() ||
                        !json::accept(call["function"]["arguments"].get<std::string>())) {
                        error = "Invalid tool call history";
                        return false;
                    }
                }
            }
            json formatted = message;
            formatted["content"] = content;
            if (formatted.contains("tool_calls")) {
                for (auto& call : formatted["tool_calls"])
                    call["function"]["arguments"] = json::parse(call["function"]["arguments"].get<std::string>());
            }
            request.messages.emplace_back("json", formatted.dump());
        } else {
            request.messages.emplace_back(role, content);
        }
    }
    return true;
}

Result generate(
    Llm* llm, const Request& request, const std::function<void(const std::string&)>& on_text,
    const std::function<void(size_t)>& on_ready = [](size_t) {}) {
    Result result;
    Utf8StreamProcessor processor([&](const std::string& text) {
        result.text += text;
        on_text(text);
    });
    LlmStreamBuffer buffer([&](const char* data, size_t size) { processor.processStream(data, size); });
    std::ostream output(&buffer);
    // Each HTTP request contains its own full history; never reuse another client's KV state.
    llm->reset();
    if (!llm->set_sampler_config(request.sampler.dump()))
        return result;
    std::vector<int> ids;
    auto rendered = llm->apply_chat_template(request.messages, request.tools.dump());
    static const std::regex media("<(img|audio|video)>(.*?)</\\1>");
    for (auto it = std::sregex_iterator(rendered.begin(), rendered.end(), media); it != std::sregex_iterator(); ++it) {
        if ((*it)[1] != "img" || request.images.count((*it)[2].str()) == 0)
            return result;
    }
    if (!request.images.empty()) {
        MNN::Transformer::MultimodalPrompt prompt;
        prompt.prompt_template = rendered;
        prompt.images = request.images;
        ids = llm->tokenizer_encode(prompt);
    } else {
        ids = llm->tokenizer_encode(rendered);
    }
    if (ids.empty())
        return result;
    on_ready(ids.size());
    llm->response(ids, &output, "", request.max_tokens);
    const auto* context = llm->getContext();
    result.prompt_tokens = context->prompt_len;
    result.completion_tokens = static_cast<int>(context->output_tokens.size());
    result.decode_tokens = result.completion_tokens;
    if (!context->output_tokens.empty() && context->status == LlmStatus::NORMAL_FINISHED)
        --result.decode_tokens;
    result.prefill_us = context->prefill_us;
    result.decode_us = context->decode_us;
    result.vision_us = context->vision_us;
    result.status = context->status;
    return result;
}

json generationTimings(const Result& result) {
    return {{"prefill_ms", result.prefill_us / 1000.0},
            {"decode_ms", result.decode_us / 1000.0},
            {"decode_tokens", result.decode_tokens},
            {"vision_ms", result.vision_us / 1000.0}};
}

bool successful(const Result& result) {
    return result.status == LlmStatus::NORMAL_FINISHED || result.status == LlmStatus::MAX_TOKENS_FINISHED;
}

std::string trimToolText(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

bool parseXmlToolCall(const std::string& text, const Request& request, json& call) {
    const auto content = trimToolText(text);
    const std::string function_tag = "<function=", function_end = "</function>", parameter_tag = "<parameter=",
                      parameter_end = "</parameter>";
    if (content.compare(0, function_tag.size(), function_tag) != 0) return false;
    const auto name_end = content.find('>', function_tag.size());
    const auto end = content.rfind(function_end);
    if (name_end == std::string::npos || end == std::string::npos || end <= name_end ||
        !trimToolText(content.substr(end + function_end.size())).empty()) return false;
    const auto name = trimToolText(content.substr(function_tag.size(), name_end - function_tag.size()));
    json properties = json::object();
    bool found = false;
    for (const auto& tool : request.tools) {
        const auto& function = tool["function"];
        if (function["name"] != name) continue;
        found = true;
        if (function.contains("parameters") && function["parameters"].is_object() &&
            function["parameters"].contains("properties") && function["parameters"]["properties"].is_object())
            properties = function["parameters"]["properties"];
        break;
    }
    if (!found) return false;
    json arguments = json::object();
    size_t pos = name_end + 1;
    while (pos < end) {
        pos = content.find_first_not_of(" \t\r\n", pos);
        if (pos == std::string::npos || pos >= end) break;
        if (content.compare(pos, parameter_tag.size(), parameter_tag) != 0) return false;
        const auto key_end = content.find('>', pos + parameter_tag.size());
        if (key_end == std::string::npos || key_end >= end) return false;
        const auto key = trimToolText(content.substr(pos + parameter_tag.size(), key_end - pos - parameter_tag.size()));
        const auto value_end = content.find(parameter_end, key_end + 1);
        if (key.empty() || value_end == std::string::npos || value_end > end || arguments.contains(key)) return false;
        const auto value = trimToolText(content.substr(key_end + 1, value_end - key_end - 1));
        const bool string_type = properties.contains(key) && properties[key].is_object() &&
                                 properties[key].contains("type") && properties[key]["type"] == "string";
        auto parsed = string_type ? json() : json::parse(value, nullptr, false);
        arguments[key] = string_type || parsed.is_discarded() ? json(value) : parsed;
        pos = value_end + parameter_end.size();
    }
    call = {{"name", name}, {"arguments", arguments}};
    return true;
}

bool parseToolCalls(Result& result, const Request& request) {
    if (request.tools.empty())
        return true;
    const std::string open = "<tool_call>", close = "</tool_call>";
    std::string remaining, text = result.text;
    size_t start = 0;
    while (true) {
        auto begin = text.find(open, start);
        if (begin == std::string::npos) {
            remaining += text.substr(start);
            break;
        }
        remaining += text.substr(start, begin - start);
        auto end = text.find(close, begin + open.size());
        if (end == std::string::npos)
            return false;
        const auto body = text.substr(begin + open.size(), end - begin - open.size());
        auto call = json::parse(body, nullptr, false);
        if (call.is_discarded() && !parseXmlToolCall(body, request, call)) return false;
        if (!call.is_object() || !call.contains("name") || !call["name"].is_string() || !call.contains("arguments"))
            return false;
        bool found = false;
        for (const auto& tool : request.tools)
            found |= tool["function"]["name"] == call["name"];
        if (!found)
            return false;
        auto arguments = call["arguments"];
        if (arguments.is_string())
            arguments = json::parse(arguments.get<std::string>(), nullptr, false);
        if (!arguments.is_object())
            return false;
        result.tool_calls.push_back({{"id", responseId("call_")},
                                     {"type", "function"},
                                     {"function", {{"name", call["name"]}, {"arguments", arguments.dump()}}}});
        start = end + close.size();
    }
    result.text = remaining;
    return true;
}

std::string finishReason(const Result& result, bool anthropic) {
    if (!result.tool_calls.empty())
        return anthropic ? "tool_use" : "tool_calls";
    if (result.status == LlmStatus::MAX_TOKENS_FINISHED)
        return anthropic ? "max_tokens" : "length";
    return anthropic ? "end_turn" : "stop";
}

} // namespace

bool MnncliServer::Start(Llm* llm, const std::string& model, const std::string& host, int port,
                         const std::string& api_key, MNN::Transformer::Embedding* embedding,
                         const std::string& embedding_model, const std::string& config_path, bool verbose) {
    httplib::Server server;
    AgentBroker agent;
    server.set_payload_max_length(16 * 1024 * 1024);
    const auto config = json::parse(llm->dump_config(), nullptr, false);
    std::string chat_template;
    if (config.is_object() && config.contains("jinja") && config["jinja"].is_object() &&
        config["jinja"].contains("chat_template") && config["jinja"]["chat_template"].is_string()) {
        chat_template = config["jinja"]["chat_template"].get<std::string>();
    }
    const bool tool_capable =
        chat_template.find("<tool_call>") != std::string::npos && chat_template.find("tools") != std::string::npos;
    const bool vision = config.is_object() && config.value("is_visual", false);
    json model_info = json::object();
    if (!config_path.empty()) {
        namespace fs = std::filesystem;
        const fs::path config_file(config_path);
        model_info["config_file"] = config_file.filename().string();
        fs::path model_dir = config_file.parent_path();
        if (config.is_object() && config.contains("base_dir") && config["base_dir"].is_string()) {
            model_dir = config["base_dir"].get<std::string>();
        }
        model_dir = model_dir.lexically_normal();
        if (model_dir.filename().empty()) model_dir = model_dir.parent_path();
        if (!model_dir.filename().empty()) model_info["directory"] = model_dir.filename().string();
    }
    if (config.is_object()) {
        for (const auto* key : {"model_type", "backend_type", "precision"}) {
            if (config.contains(key) && config[key].is_string() && !config[key].get<std::string>().empty())
                model_info[key] = config[key];
        }
        for (const auto* key : {"thread_num", "hidden_size", "layer_nums"}) {
            if (config.contains(key) && config[key].is_number_integer() && config[key].get<int64_t>() > 0)
                model_info[key] = config[key];
        }
    }
    const auto created = static_cast<int64_t>(std::time(nullptr));
    auto authorized = [&](const httplib::Request& req, httplib::Response& res, bool anthropic) {
        if (api_key.empty() || req.get_header_value("Authorization") == "Bearer " + api_key ||
            req.get_header_value("x-api-key") == api_key)
            return true;
        sendError(res, 401, "Invalid API key", anthropic);
        return false;
    };
    auto localToolAuthorized = [&](const httplib::Request& req, httplib::Response& res) {
        const auto host_header = req.get_header_value("Host");
        const auto suffix = ":" + std::to_string(port);
        const bool local_host = host_header == "127.0.0.1" + suffix || host_header == "localhost" + suffix ||
                                (port == 80 && (host_header == "127.0.0.1" || host_header == "localhost"));
        if (req.remote_addr != "127.0.0.1" || !local_host || req.get_header_value("Origin") != "http://" + host_header) {
            sendError(res, 403, "Local tools require a same-origin local request", false);
            return false;
        }
        return authorized(req, res, false);
    };
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        allowCors(res);
        res.set_content(mnncli_web_ui, "text/html");
    });
    server.Get("/api/status", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        if (!authorized(req, res, false)) return;
        json status = {{"model", model}, {"vision", vision}, {"tools", tool_capable},
                       {"embedding_model", embedding ? json(embedding_model) : json(nullptr)}, {"model_info", model_info}};
#ifdef _WIN32
        status["agent_tools"] = false;
#else
        status["agent_tools"] = req.remote_addr == "127.0.0.1";
#endif
        res.set_content(status.dump(), "application/json");
    });
    server.Post("/api/agent/session", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        const auto body = json::parse(req.body, nullptr, false);
        json result;
        std::string error;
        if (!agent.openSession(body, result, error)) { sendError(res, 400, error, false); return; }
        res.set_content(result.dump(), "application/json");
    });
    server.Post("/api/agent/close", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        const auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("session_id") || !body["session_id"].is_string()) {
            sendError(res, 400, "Invalid agent session", false); return;
        }
        agent.closeSession(body["session_id"].get<std::string>());
        res.set_content("{}", "application/json");
    });
    server.Post("/api/agent/prepare", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("session_id") || !body["session_id"].is_string()) {
            sendError(res, 400, "Invalid agent session", false); return;
        }
        json result;
        std::string error;
        if (!agent.prepare(body["session_id"].get<std::string>(), body, result, error)) { sendError(res, 400, error, false); return; }
        res.set_content(result.dump(), "application/json");
    });
    server.Post("/api/agent/execute", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("session_id") || !body["session_id"].is_string() || !body.contains("request_id") || !body["request_id"].is_string() || body["request_id"].get<std::string>().size() != 48) {
            sendError(res, 400, "Invalid approval request ID", false); return;
        }
        json result;
        std::string error;
        if (!agent.execute(body["session_id"].get<std::string>(), body["request_id"].get<std::string>(), result, error)) { sendError(res, 400, error, false); return; }
        res.set_content(result.dump(), "application/json");
    });
    server.Post("/api/agent/cancel", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("session_id") || !body["session_id"].is_string() || !body.contains("request_id") || !body["request_id"].is_string()) {
            sendError(res, 400, "Invalid approval request ID", false); return;
        }
        agent.cancel(body["session_id"].get<std::string>(), body["request_id"].get<std::string>());
        res.set_content("{}", "application/json");
    });
    server.Post("/api/agent/mcp/disconnect", [&](const httplib::Request& req, httplib::Response& res) {
        if (!localToolAuthorized(req, res)) return;
        const auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("session_id") || !body["session_id"].is_string() ||
            !body.contains("server") || !body["server"].is_string()) {
            sendError(res, 400, "Invalid agent session or MCP server", false); return;
        }
        std::string error;
        if (!agent.disconnectMcp(body["session_id"].get<std::string>(), body["server"].get<std::string>(), error)) {
            sendError(res, 400, error, false); return;
        }
        res.set_content("{}", "application/json");
    });
    server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        if (!authorized(req, res, false))
            return;
        json data = json::array({{{"id", model}, {"object", "model"}, {"created", created}, {"owned_by", "mnn"}}});
        if (embedding)
            data.push_back({{"id", embedding_model}, {"object", "model"}, {"created", created}, {"owned_by", "mnn"}});
        res.set_content(json({{"object", "list"}, {"data", data}}).dump(), "application/json");
    });
    server.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        if (!authorized(req, res, false))
            return;
        const auto body = json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("model") || !body["model"].is_string()) {
            sendError(res, 400, "model and input are required", false);
            return;
        }
        if (!embedding || body["model"].get<std::string>() != embedding_model) {
            sendError(res, 404, "Embedding model not found", false);
            return;
        }
        if ((body.contains("encoding_format") && !body["encoding_format"].is_null() &&
             body["encoding_format"] != "float") ||
            (body.contains("dimensions") && !body["dimensions"].is_null())) {
            sendError(res, 400, "Only float embeddings with the model's native dimensions are supported", false);
            return;
        }
        if (!body.contains("input")) {
            sendError(res, 400, "input is required", false);
            return;
        }
        json inputs = body["input"].is_string() ? json::array({body["input"]}) : body["input"];
        if (!inputs.is_array() || inputs.empty() || inputs.size() > 128) {
            sendError(res, 400, "input must be a string or a non-empty array of up to 128 strings", false);
            return;
        }
        for (const auto& item : inputs) {
            if (!item.is_string() || item.get<std::string>().empty()) {
                sendError(res, 400, "Embedding inputs must be non-empty strings", false);
                return;
            }
        }
        std::lock_guard<std::mutex> lock(llm_mutex_);
        json data = json::array();
        int tokens = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto vector = embedding->txt_embedding(inputs[i].get<std::string>());
            if (!vector.get() || !vector->getInfo() || !vector->readMap<float>()) {
                sendError(res, 500, "Embedding inference failed", false);
                return;
            }
            const auto size = vector->getInfo()->size;
            const float* values = vector->readMap<float>();
            std::vector<float> result(values, values + size);
            double squared_norm = 0;
            for (const auto value : result)
                squared_norm += static_cast<double>(value) * value;
            if (!std::isfinite(squared_norm) || squared_norm <= 0) {
                sendError(res, 500, "Embedding contains invalid values", false);
                return;
            }
            const double norm = std::sqrt(squared_norm);
            for (auto& value : result)
                value = static_cast<float>(value / norm);
            tokens += embedding->getContext()->prompt_len;
            data.push_back({{"object", "embedding"}, {"index", i}, {"embedding", result}});
        }
        res.set_content(json({{"object", "list"},
                              {"data", data},
                              {"model", embedding_model},
                              {"usage", {{"prompt_tokens", tokens}, {"total_tokens", tokens}}}})
                            .dump(),
                        "application/json");
    });
    server.Post("/reset", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        if (!authorized(req, res, false))
            return;
        std::lock_guard<std::mutex> lock(llm_mutex_);
        llm->reset();
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    auto handler = [&](const httplib::Request& req, httplib::Response& res, bool anthropic) {
        if (anthropic && verbose) std::fputs("POST /v1/messages\n", stderr);
        allowCors(res);
        if (!authorized(req, res, anthropic))
            return;
        const auto body = json::parse(req.body, nullptr, false);
        Request request;
        std::string error;
        if (body.is_discarded()) {
            sendError(res, 400, "Invalid JSON", anthropic);
            return;
        }
        bool valid_request;
        {
            std::lock_guard<std::mutex> lock(llm_mutex_);
            MNN::Express::ExecutorScope scope(llm->getExecutor());
            valid_request = parseRequest(body, model, anthropic, tool_capable, vision, request, error);
        }
        if (!valid_request) {
            const bool missing_model = body.is_object() && body.contains("model") && body["model"].is_string() &&
                                       body["model"].get<std::string>() != model;
            sendError(res, missing_model ? 404 : 400, error, anthropic);
            return;
        }
        const auto id = responseId(anthropic ? "msg_" : "chatcmpl-");
        const auto timestamp = static_cast<int64_t>(std::time(nullptr));
        if (!request.stream) {
            std::lock_guard<std::mutex> lock(llm_mutex_);
            auto result = generate(llm, request, [](const std::string&) {});
            if (!successful(result) || !parseToolCalls(result, request)) {
                sendError(res, 500, "Model generation failed", anthropic);
                return;
            }
            json response;
            if (anthropic) {
                json content = json::array();
                if (!result.text.empty() || result.tool_calls.empty())
                    content.push_back({{"type", "text"}, {"text", result.text}});
                for (const auto& call : result.tool_calls) {
                    content.push_back({{"type", "tool_use"},
                                       {"id", call["id"]},
                                       {"name", call["function"]["name"]},
                                       {"input", json::parse(call["function"]["arguments"].get<std::string>())}});
                }
                response = {
                    {"id", id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", model},
                    {"content", content},
                    {"stop_reason", finishReason(result, true)},
                    {"stop_sequence", nullptr},
                    {"usage", {{"input_tokens", result.prompt_tokens}, {"output_tokens", result.completion_tokens}}}};
            } else {
                json message = {
                    {"role", "assistant"},
                    {"content", result.text.empty() && !result.tool_calls.empty() ? json(nullptr) : json(result.text)}};
                if (!result.tool_calls.empty())
                    message["tool_calls"] = result.tool_calls;
                response = {{"id", id},
                            {"object", "chat.completion"},
                            {"created", timestamp},
                            {"model", model},
                            {"choices", json::array({{{"index", 0},
                                                      {"message", message},
                                                      {"finish_reason", finishReason(result, false)}}})},
                            {"usage",
                             {{"prompt_tokens", result.prompt_tokens},
                              {"completion_tokens", result.completion_tokens},
                              {"total_tokens", result.prompt_tokens + result.completion_tokens}}},
                            {"mnn_timings", generationTimings(result)}};
            }
            res.set_content(response.dump(), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [=](size_t, httplib::DataSink& sink) {
            std::lock_guard<std::mutex> lock(llm_mutex_);
            auto openaiChunk = [&](const json& delta, const json& reason) {
                sendEvent(sink, "",
                          {{"id", id},
                           {"object", "chat.completion.chunk"},
                           {"created", timestamp},
                           {"model", model},
                           {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", reason}}})}});
            };
            if (!anthropic)
                openaiChunk({{"role", "assistant"}}, nullptr);
            auto ready = [&](size_t input_tokens) {
                if (!anthropic)
                    return;
                sendEvent(sink, "message_start",
                          {{"type", "message_start"},
                           {"message",
                            {{"id", id},
                             {"type", "message"},
                             {"role", "assistant"},
                             {"model", model},
                             {"content", json::array()},
                             {"stop_reason", nullptr},
                             {"stop_sequence", nullptr},
                             {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}}}}});
                if (request.tools.empty()) {
                    sendEvent(sink, "content_block_start",
                              {{"type", "content_block_start"},
                               {"index", 0},
                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
            };
            auto result = generate(
                llm, request,
                [&](const std::string& text) {
                    if (!request.tools.empty())
                        return;
                    if (anthropic)
                        sendEvent(sink, "content_block_delta",
                                  {{"type", "content_block_delta"},
                                   {"index", 0},
                                   {"delta", {{"type", "text_delta"}, {"text", text}}}});
                    else
                        openaiChunk({{"content", text}}, nullptr);
                },
                ready);
            if (!successful(result) || !parseToolCalls(result, request)) {
                sendEvent(
                    sink, anthropic ? "error" : "",
                    {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", "Model generation failed"}}}});
            } else if (anthropic) {
                size_t index = 0;
                if (request.tools.empty()) {
                    sendEvent(sink, "content_block_stop", {{"type", "content_block_stop"}, {"index", index++}});
                } else if (!result.text.empty() || result.tool_calls.empty()) {
                    sendEvent(sink, "content_block_start",
                              {{"type", "content_block_start"},
                               {"index", index},
                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                    if (!result.text.empty())
                        sendEvent(sink, "content_block_delta",
                                  {{"type", "content_block_delta"},
                                   {"index", index},
                                   {"delta", {{"type", "text_delta"}, {"text", result.text}}}});
                    sendEvent(sink, "content_block_stop", {{"type", "content_block_stop"}, {"index", index++}});
                }
                for (const auto& call : result.tool_calls) {
                    sendEvent(sink, "content_block_start",
                              {{"type", "content_block_start"},
                               {"index", index},
                               {"content_block",
                                {{"type", "tool_use"},
                                 {"id", call["id"]},
                                 {"name", call["function"]["name"]},
                                 {"input", json::object()}}}});
                    sendEvent(
                        sink, "content_block_delta",
                        {{"type", "content_block_delta"},
                         {"index", index},
                         {"delta", {{"type", "input_json_delta"}, {"partial_json", call["function"]["arguments"]}}}});
                    sendEvent(sink, "content_block_stop", {{"type", "content_block_stop"}, {"index", index++}});
                }
                sendEvent(
                    sink, "message_delta",
                    {{"type", "message_delta"},
                     {"delta", {{"stop_reason", finishReason(result, true)}, {"stop_sequence", nullptr}}},
                     {"usage", {{"input_tokens", result.prompt_tokens}, {"output_tokens", result.completion_tokens}}}});
                sendEvent(sink, "message_stop", {{"type", "message_stop"}});
            } else {
                if (!request.tools.empty() && !result.text.empty())
                    openaiChunk({{"content", result.text}}, nullptr);
                for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                    auto call = result.tool_calls[i];
                    call["index"] = i;
                    openaiChunk({{"tool_calls", json::array({call})}}, nullptr);
                }
                openaiChunk(json::object(), finishReason(result, false));
                if (request.include_usage) {
                    sendEvent(sink, "",
                              {{"id", id},
                               {"object", "chat.completion.chunk"},
                               {"created", timestamp},
                               {"model", model},
                               {"choices", json::array()},
                               {"usage",
                                {{"prompt_tokens", result.prompt_tokens},
                                 {"completion_tokens", result.completion_tokens},
                                 {"total_tokens", result.prompt_tokens + result.completion_tokens}}},
                               {"mnn_timings", generationTimings(result)}});
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
            }
            sink.done();
            return true;
        });
    };
    server.Post("/v1/chat/completions",
                [&](const httplib::Request& req, httplib::Response& res) { handler(req, res, false); });
    server.Post("/chat/completions",
                [&](const httplib::Request& req, httplib::Response& res) { handler(req, res, false); });
    server.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) { handler(req, res, true); });
    for (const auto* path :
         {"/api/status", "/v1/models", "/v1/chat/completions", "/chat/completions", "/v1/messages", "/v1/embeddings", "/reset"}) {
        server.Options(path, [](const httplib::Request&, httplib::Response& res) {
            allowCors(res);
            res.status = 200;
        });
    }
    if (!server.listen(host.c_str(), port)) {
        std::fprintf(stderr, "Could not start server on %s:%d\n", host.c_str(), port);
        return false;
    }
    return true;
}
} // namespace mnncli
