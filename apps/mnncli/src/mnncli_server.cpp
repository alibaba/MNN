//
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mnncli_server.hpp"
#include "chat_page.hpp"
#include "omni.hpp"
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <regex>
#include <set>
#include <cv/imgcodecs.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <openssl/evp.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "imageHelper/stb_image.h"

namespace mnncli {
namespace {

using MNN::Transformer::Llm;
using MNN::Transformer::LlmStatus;
using MNN::Transformer::MultimodalPrompt;

struct Request {
    MultimodalPrompt prompt;
    json sampler = json::object();
    bool stream = false;
    bool include_usage = false;
    int max_tokens = -1;
};

struct Result {
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
    if (data.empty() || data.size() > 12 * 1024 * 1024 || data.size() % 4 != 0 || request.prompt.images.size() >= 8) {
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
    int width, height, channels;
    if (!stbi_info_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height, &channels) ||
        width <= 0 || height <= 0 || width > (16 * 1024 * 1024) / height) {
        error = "Invalid image dimensions or image exceeds the pixel limit";
        return false;
    }
    auto image = MNN::CV::imdecode(bytes, MNN::CV::IMREAD_COLOR);
    if (!image.get() || !image->getInfo() || image->getInfo()->dim.size() != 3) {
        error = "Could not decode image";
        return false;
    }
    const auto& dims = image->getInfo()->dim;
    const auto key = "mnncli-image-" + std::to_string(request.prompt.images.size());
    request.prompt.images.emplace(key, MNN::Transformer::PromptImagePart{image, dims[1], dims[0]});
    // Keep a literal opening tag in preceding text from consuming this image.
    text += "\n<img>" + key + "</img>";
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
                text += part["text"].get<std::string>();
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
    return true;
}

bool validMediaReferences(const MultimodalPrompt& prompt) {
    // Omni resolves unmatched image keys and all audio/video references as files.
    static const std::regex media(MNN::Transformer::kOmniMultimodalRegex);
    const auto& text = prompt.prompt_template;
    std::set<std::string> used;
    for (std::sregex_iterator it(text.begin(), text.end(), media), end; it != end; ++it) {
        const auto key = (*it)[2].str();
        if ((*it)[1] != "img" || !prompt.images.count(key) || !used.insert(key).second)
            return false;
    }
    return true;
}

bool parseRequest(Llm* llm, const json& body, const std::string& model, bool anthropic, bool vision, Request& request,
                  std::string& error) {
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
    const char* unsupported[] = {"stop",
                                 "stop_sequences",
                                 "response_format",
                                 "seed",
                                 "presence_penalty",
                                 "frequency_penalty",
                                 "logit_bias",
                                 "logprobs",
                                 "top_logprobs",
                                 "service_tier",
                                 "thinking",
                                 "output_config",
                                 "tool_choice",
                                 "parallel_tool_calls",
                                 "functions",
                                 "function_call"};
    for (const auto* field : unsupported) {
        if (body.contains(field) && !body[field].is_null()) {
            error = std::string("Unsupported parameter: ") + field;
            return false;
        }
    }
    if (body.contains("tools") && !body["tools"].is_null() && (!body["tools"].is_array() || !body["tools"].empty())) {
        error = "Tool calling is not supported";
        return false;
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
        if (!request.sampler.contains("temperature"))
            request.sampler["temperature"] = 1.0;
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
    std::vector<PromptItem> messages;
    if (anthropic && body.contains("system")) {
        std::string system;
        if (!textContent(body["system"], system)) {
            error = "Unsupported system content";
            return false;
        }
        messages.emplace_back("system", system);
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
        if (role != "user" && role != "assistant" && (anthropic || role != "system")) {
            error = "Unsupported message role: " + role;
            return false;
        }
        if ((message.contains("tool_calls") && !message["tool_calls"].is_null()) ||
            (message.contains("function_call") && !message["function_call"].is_null())) {
            error = "Tool call history is not supported";
            return false;
        }
        std::string content;
        if (!messageContent(message["content"], anthropic, vision, request, content, error))
            return false;
        messages.emplace_back(role, content);
    }
    request.prompt.prompt_template = llm->apply_chat_template(messages);
    if (!validMediaReferences(request.prompt)) {
        error = "Raw media references are not supported; use image content blocks";
        return false;
    }
    return true;
}

json samplerDefaults(const std::string& model_config) {
    MNN::Transformer::LlmConfig config;
    config.config_ = ujson::json::parse(model_config);
    return {{"sampler_type", config.sampler_type()},
            {"mixed_samplers", config.mixed_samplers()},
            {"temperature", config.temperature()},
            {"top_k", config.topK()},
            {"top_p", config.topP()}};
}

Result generate(
    Llm* llm, const Request& request, const json& defaults, const std::function<void(const std::string&)>& on_text,
    const std::function<void(size_t)>& on_ready = [](size_t) {}) {
    Result result;
    Utf8StreamProcessor processor(on_text);
    LlmStreamBuffer buffer([&](const char* data, size_t size) { processor.processStream(data, size); });
    std::ostream output(&buffer);
    // Each HTTP request contains its own full history; never reuse another client's KV state.
    llm->reset();
    auto settings = defaults;
    settings.update(request.sampler);
    if (!llm->set_config(settings.dump()))
        return result;
    auto ids = llm->tokenizer_encode(request.prompt);
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

std::string finishReason(const Result& result, bool anthropic) {
    if (result.status == LlmStatus::MAX_TOKENS_FINISHED)
        return anthropic ? "max_tokens" : "length";
    return anthropic ? "end_turn" : "stop";
}

} // namespace

bool MnncliServer::Start(Llm* llm, const std::string& model, const std::string& host, int port,
                         const std::string& api_key, MNN::Transformer::Embedding* embedding,
                         const std::string& embedding_model, const std::string& config_path, bool verbose) {
    httplib::Server server;
    server.set_payload_max_length(16 * 1024 * 1024);
    const auto model_config = llm->dump_config();
    const auto defaults = samplerDefaults(model_config);
    const auto config = json::parse(model_config, nullptr, false);
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
        if (model_dir.filename().empty())
            model_dir = model_dir.parent_path();
        if (!model_dir.filename().empty())
            model_info["directory"] = model_dir.filename().string();
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
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(kChatPage, "text/html; charset=utf-8");
    });
    server.Get("/api/status", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        if (!authorized(req, res, false))
            return;
        json status = {{"model", model},
                       {"vision", vision},
                       {"embedding_model", embedding ? json(embedding_model) : json(nullptr)},
                       {"model_info", model_info}};
        res.set_content(status.dump(), "application/json");
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
            MultimodalPrompt prompt;
            prompt.prompt_template = embedding->apply_chat_template(inputs[i].get<std::string>());
            if (!validMediaReferences(prompt)) {
                sendError(res, 400, "Embedding inputs must not reference media", false);
                return;
            }
            auto vector = embedding->ids_embedding(embedding->tokenizer_encode(prompt));
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
        if (anthropic && verbose)
            std::fputs("POST /v1/messages\n", stderr);
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
            valid_request = parseRequest(llm, body, model, anthropic, vision, request, error);
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
            std::string text;
            auto result = generate(llm, request, defaults, [&](const std::string& chunk) { text += chunk; });
            if (!successful(result)) {
                sendError(res, 500, "Model generation failed", anthropic);
                return;
            }
            json response;
            if (anthropic) {
                response = {
                    {"id", id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", model},
                    {"content", json::array({{{"type", "text"}, {"text", text}}})},
                    {"stop_reason", finishReason(result, true)},
                    {"stop_sequence", nullptr},
                    {"usage", {{"input_tokens", result.prompt_tokens}, {"output_tokens", result.completion_tokens}}}};
            } else {
                response = {{"id", id},
                            {"object", "chat.completion"},
                            {"created", timestamp},
                            {"model", model},
                            {"choices", json::array({{{"index", 0},
                                                      {"message", {{"role", "assistant"}, {"content", text}}},
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
                sendEvent(sink, "content_block_start",
                          {{"type", "content_block_start"},
                           {"index", 0},
                           {"content_block", {{"type", "text"}, {"text", ""}}}});
            };
            auto result = generate(
                llm, request, defaults,
                [&](const std::string& text) {
                    if (anthropic)
                        sendEvent(sink, "content_block_delta",
                                  {{"type", "content_block_delta"},
                                   {"index", 0},
                                   {"delta", {{"type", "text_delta"}, {"text", text}}}});
                    else
                        openaiChunk({{"content", text}}, nullptr);
                },
                ready);
            if (!successful(result)) {
                sendEvent(
                    sink, anthropic ? "error" : "",
                    {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", "Model generation failed"}}}});
            } else if (anthropic) {
                sendEvent(sink, "content_block_stop", {{"type", "content_block_stop"}, {"index", 0}});
                sendEvent(
                    sink, "message_delta",
                    {{"type", "message_delta"},
                     {"delta", {{"stop_reason", finishReason(result, true)}, {"stop_sequence", nullptr}}},
                     {"usage", {{"input_tokens", result.prompt_tokens}, {"output_tokens", result.completion_tokens}}}});
                sendEvent(sink, "message_stop", {{"type", "message_stop"}});
            } else {
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
    for (const auto* path : {"/api/status", "/v1/models", "/v1/chat/completions", "/chat/completions", "/v1/messages",
                             "/v1/embeddings", "/reset"}) {
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
