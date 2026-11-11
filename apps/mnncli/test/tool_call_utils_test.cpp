#include "tool_call_utils.hpp"
#include <cassert>
#include <iostream>
using namespace mnncli::toolcall;
int main() {
    auto tools = json::parse(R"([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string"},"count":{"type":"integer"}},"required":["city"]}}}])");
    json message;
    std::string error;
    assert(parse("<tool_call>\n<function=lookup>\n<parameter=city>\n北京\n</parameter>\n<parameter=count>2</parameter>\n</function>\n</tool_call>", tools, "1", message, error));
    assert(json::parse(message["tool_calls"][0]["function"]["arguments"].get<std::string>())["city"] == "北京");
    assert(json::parse(message["tool_calls"][0]["function"]["arguments"].get<std::string>())["count"] == 2);
    assert(parse("<think>reason</think><tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"上海\"}}</tool_call>", tools, "2", message, error));
    assert(message["content"].is_null());
    assert(!parse("<tool_call><function=lookup>", tools, "3", message, error));
    assert(!parse("<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>", tools, "4", message, error));
    assert(!parse("<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>", tools, "5", message, error));
    assert(parse("hello", tools, "6", message, error) && message["content"] == "hello");
    assert(parse("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"北京\"}}</tool_call><tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"天津\"}}</tool_call>", tools, "7", message, error));
    assert(message["tool_calls"].size() == 2);
    json request = {{"tools", tools}, {"messages", json::array({{{"role", "user"}, {"content", "天气"}}, message})}};
    json normalized, supplied;
    assert(!prepare(request, normalized, supplied, error));
    for (const auto& call : message["tool_calls"]) request["messages"].push_back({{"role", "tool"}, {"tool_call_id", call["id"]}, {"content", "晴"}});
    assert(prepare(request, normalized, supplied, error));
    assert(normalized[1]["tool_calls"][0]["function"]["arguments"].is_object());
    assert(normalized[1]["content"] == "");
    request["tool_choice"] = "none";
    assert(prepare(request, normalized, supplied, error) && supplied.empty());
    request["tool_choice"] = "required";
    assert(!prepare(request, normalized, supplied, error));
    request["tool_choice"] = "auto";
    request["messages"][2]["tool_call_id"] = "wrong";
    assert(!prepare(request, normalized, supplied, error));
    assert(parse("<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"</think>\"}}</tool_call>", tools, "8", message, error));
    assert(json::parse(message["tool_calls"][0]["function"]["arguments"].get<std::string>())["city"] == "</think>");
    const std::string example = "example <tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"北京\"}}</tool_call>";
    assert(parse(example, tools, "9", message, error, true));
    assert(!message.contains("tool_calls"));
    assert(parse(example + "</think>answer", tools, "10", message, error, true));
    assert(message["content"] == "answer" && !message.contains("tool_calls"));
    std::cout << "PASS: XML/JSON, typed arguments, malformed/unknown/missing arguments, multiple calls, roundtrip/null, none/required, unmatched id\n";
}
