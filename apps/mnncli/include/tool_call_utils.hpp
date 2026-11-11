#pragma once

#include "json.hpp"
#include <string>
#include <set>

namespace mnncli {
namespace toolcall {
using nlohmann::json;

inline std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

inline const json* function(const json& tools, const std::string& name) {
    for (const auto& tool : tools) {
        if (tool["function"]["name"] == name) return &tool["function"];
    }
    return nullptr;
}

inline bool prepare(const json& request, json& messages, json& tools, std::string& error) {
    auto fail = [&](const char* reason) { error = reason; return false; };
    tools = request.value("tools", json::array());
    if (!tools.is_array()) return fail("tools must be an array");
    std::set<std::string> names;
    for (const auto& tool : tools) {
        if (!tool.is_object() || !tool.contains("type") || tool["type"] != "function" ||
            !tool.contains("function") || !tool["function"].is_object()) {
            return fail("Only function tools are supported");
        }
        const auto& fn = tool["function"];
        if (!fn.contains("name") || !fn["name"].is_string() || fn["name"].get<std::string>().empty() ||
            !names.insert(fn["name"].get<std::string>()).second) return fail("Invalid or duplicate tool name");
        if (fn.contains("parameters") && !fn["parameters"].is_object()) return fail("Invalid tool parameters");
    }
    const auto choice = request.value("tool_choice", json("auto"));
    if (choice != "auto" && choice != "none") {
        return fail("Supported tool_choice values are auto and none; forced tool calls are not supported");
    }
    if (!request.contains("messages") || !request["messages"].is_array() || request["messages"].empty()) {
        return fail("messages must be a non-empty array");
    }
    messages = request["messages"];
    std::set<std::string> pending;
    for (auto& message : messages) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string()) {
            return fail("Invalid message role");
        }
        const auto role = message["role"].get<std::string>();
        if (role != "system" && role != "user" && role != "assistant" && role != "tool") {
            return fail("Unsupported message role");
        }
        if (!pending.empty() && role != "tool") return fail("Missing tool results before next message");
        if (!message.contains("content") || message["content"].is_null()) {
            if (role != "assistant" || !message.contains("tool_calls")) return fail("Missing message content");
            message["content"] = "";
        }
        if (!message["content"].is_string()) return fail("Tool requests currently require text message content");
        if (message.contains("tool_calls")) {
            if (role != "assistant" || !message["tool_calls"].is_array() || message["tool_calls"].empty()) {
                return fail("Invalid assistant tool_calls");
            }
            for (auto& call : message["tool_calls"]) {
                if (!call.is_object() || !call.contains("id") || !call["id"].is_string() ||
                    call["id"].get<std::string>().empty() || !call.contains("type") || call["type"] != "function" ||
                    !call.contains("function") || !call["function"].is_object()) return fail("Invalid tool call");
                auto& fn = call["function"];
                if (!fn.contains("name") || !fn["name"].is_string() || !fn.contains("arguments")) {
                    return fail("Invalid tool call function");
                }
                if (fn["arguments"].is_string()) {
                    fn["arguments"] = json::parse(fn["arguments"].get<std::string>(), nullptr, false);
                }
                if (!fn["arguments"].is_object()) return fail("Tool arguments must encode a JSON object");
                if (!pending.insert(call["id"].get<std::string>()).second) return fail("Duplicate tool call id");
            }
        }
        if (role == "tool") {
            if (!message.contains("tool_call_id") || !message["tool_call_id"].is_string() ||
                !pending.erase(message["tool_call_id"].get<std::string>())) return fail("Unmatched tool_call_id");
        }
    }
    if (!pending.empty()) return fail("Missing tool results");
    if (choice == "none") tools = json::array();
    if (request.contains("max_tokens") && (!request["max_tokens"].is_number_integer() ||
        request["max_tokens"].get<int64_t>() <= 0 || request["max_tokens"].get<int64_t>() > 32768)) {
        return fail("max_tokens must be between 1 and 32768");
    }
    return true;
}

// Qwen3.5 emits XML-style function blocks; older Qwen templates emit JSON blocks.
inline bool parseBlock(const std::string& block, const json& tools, json& fn) {
    const auto body = trim(block);
    if (body.empty()) return false;
    if (body.front() == '{') {
        fn = json::parse(body, nullptr, false);
        if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string() ||
            !fn.contains("arguments")) return false;
        if (fn["arguments"].is_string()) fn["arguments"] = json::parse(fn["arguments"].get<std::string>(), nullptr, false);
    } else {
        if (body.compare(0, 10, "<function=") != 0) return false;
        const auto nameEnd = body.find('>', 10);
        const auto close = body.rfind("</function>");
        if (nameEnd == std::string::npos || close == std::string::npos || close < nameEnd ||
            !trim(body.substr(close + 11)).empty()) return false;
        const auto name = body.substr(10, nameEnd - 10);
        const auto* spec = function(tools, name);
        if (!spec) return false;
        fn = {{"name", name}, {"arguments", json::object()}};
        size_t pos = nameEnd + 1;
        while (pos < close) {
            const auto start = body.find("<parameter=", pos);
            if (start == std::string::npos || start >= close) return trim(body.substr(pos, close - pos)).empty();
            if (!trim(body.substr(pos, start - pos)).empty()) return false;
            const auto keyEnd = body.find('>', start + 11);
            const auto valueEnd = body.find("</parameter>", keyEnd);
            if (keyEnd == std::string::npos || valueEnd == std::string::npos || valueEnd >= close) return false;
            const auto key = body.substr(start + 11, keyEnd - start - 11);
            if (key.empty() || fn["arguments"].contains(key)) return false;
            const auto value = trim(body.substr(keyEnd + 1, valueEnd - keyEnd - 1));
            json decoded = value;
            if (spec->contains("parameters") && (*spec)["parameters"].contains("properties")) {
                const auto& props = (*spec)["parameters"]["properties"];
                if (props.is_object() && props.contains(key) && props[key].is_object() &&
                    props[key].contains("type") && props[key]["type"] != "string") {
                    decoded = json::parse(value, nullptr, false);
                    if (decoded.is_discarded()) return false;
                }
            }
            fn["arguments"][key] = decoded;
            pos = valueEnd + 12;
        }
    }
    return fn.is_object() && fn.contains("arguments") && fn["arguments"].is_object() &&
           function(tools, fn["name"].get<std::string>()) != nullptr;
}

inline bool parse(const std::string& output, const json& tools, const std::string& id,
                  json& message, std::string& error, bool thinking = false) {
    std::string text = output;
    if (thinking || trim(text).compare(0, 7, "<think>") == 0) {
        const auto think = text.find("</think>");
        // A token limit can leave reasoning unfinished; never interpret its examples as calls.
        if (think == std::string::npos) {
            message = {{"role", "assistant"}, {"content", text}};
            return true;
        }
        text.erase(0, think + 8);
    }
    json calls = json::array();
    std::string content;
    size_t pos = 0;
    while (true) {
        const auto start = text.find("<tool_call>", pos);
        if (start == std::string::npos) { content += text.substr(pos); break; }
        content += text.substr(pos, start - pos);
        const auto end = text.find("</tool_call>", start + 11);
        json fn;
        if (end == std::string::npos || !parseBlock(text.substr(start + 11, end - start - 11), tools, fn)) {
            error = "Model generated an incomplete or invalid tool call";
            return false;
        }
        const auto* spec = function(tools, fn["name"].get<std::string>());
        if (spec && spec->contains("parameters") && (*spec)["parameters"].contains("required")) {
            const auto& required = (*spec)["parameters"]["required"];
            if (!required.is_array()) { error = "Invalid required parameter schema"; return false; }
            for (const auto& key : required) {
                if (!key.is_string() || !fn["arguments"].contains(key.get<std::string>())) {
                    error = "Model omitted a required tool argument"; return false;
                }
            }
        }
        fn["arguments"] = fn["arguments"].dump();
        calls.push_back({{"id", "call_" + id + "_" + std::to_string(calls.size())},
                         {"type", "function"}, {"function", fn}});
        pos = end + 12;
    }
    message = {{"role", "assistant"}, {"content", trim(content)}};
    if (!calls.empty()) {
        if (trim(content).empty()) message["content"] = nullptr;
        message["tool_calls"] = calls;
    }
    return true;
}
} // namespace toolcall
} // namespace mnncli
