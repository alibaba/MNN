/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include "rapidjson/document.h"

// Forward declaration for Value used in Minja
namespace minja {
class Value;
class TemplateNode;
}

using Document = rapidjson::Document;
// Note: rapidjson::Value is the type for all JSON values (objects, arrays, strings, numbers, booleans, null).
// rapidjson::Document inherits from rapidjson::Value and holds the memory allocations for the DOM.
// We will use rapidjson::Value where nlohmann::json was used for individual values,
// and rapidjson::Document where a new JSON structure was being parsed or built.

namespace minja {

struct chat_template_caps {
    bool supports_tools = false;
    bool supports_tool_calls = false;
    bool supports_tool_responses = false;
    bool supports_system_role = false;
    bool supports_parallel_tool_calls = false;
    bool supports_tool_call_id = false;
    // meta-llama/Llama-3.1-8B-Instruct expects arguments to be an object.
    // Most other templates (and OpenAI's API) expect the arguments object to be stringified.
    bool requires_object_arguments = false;
    // CohereForAI/c4ai-command-r-plus simple variant
    bool requires_non_null_content = false;
    // MiniMaxAI/MiniMax-Text-01 special
    bool requires_typed_content = false;
};

struct chat_template_inputs {
    rapidjson::Document messages; // Should be an array
    rapidjson::Document tools;    // Should be an array or null
    bool add_generation_prompt = true;
    rapidjson::Document extra_context; // Should be an object or null
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    rapidjson::Document::AllocatorType* allocator_for_inputs = nullptr; // To be set when creating inputs
    
    // Default constructor to initialize Value members
    chat_template_inputs() : messages(rapidjson::kArrayType), tools(rapidjson::kNullType), extra_context(rapidjson::kNullType) {}
};

struct chat_template_options {
    bool apply_polyfills = true;
    bool use_bos_token = true;
    bool use_eos_token = true;
    bool define_strftime_now = true;
    
    bool polyfill_tools = true;
    bool polyfill_tool_call_examples = true;
    bool polyfill_tool_calls = true;
    bool polyfill_tool_responses = true;
    bool polyfill_system_role = true;
    bool polyfill_object_arguments = true;
    bool polyfill_typed_content = true;
};

class chat_template {
    
private:
    chat_template_caps caps_;
    std::string source_;
    std::string bos_token_;
    std::string eos_token_;
    std::shared_ptr<minja::TemplateNode> template_root_;
    std::string tool_call_example_;
    
    std::string try_raw_render(
                               rapidjson::Value& messages, // Modifying to pass by ref as it might be changed by polyfills later
                               rapidjson::Value& tools,    // Modifying to pass by ref
                               bool add_generation_prompt,
                               rapidjson::Document::AllocatorType& allocator, // Added allocator
                               rapidjson::Value extra_context = rapidjson::Value(rapidjson::kNullType)) const; // Default to null
public:
    MNN_PUBLIC chat_template(const std::string & source, const std::string & bos_token, const std::string & eos_token);
    
    const std::string & source() const { return source_; }
    const std::string & bos_token() const { return bos_token_; }
    const std::string & eos_token() const { return eos_token_; }
    const chat_template_caps & original_caps() const { return caps_; }
    
    
    MNN_PUBLIC std::string apply(
                      chat_template_inputs & inputs,
                      const chat_template_options & opts = chat_template_options()) const;
    
    static rapidjson::Value add_system(
                                       const rapidjson::Value & messages_const, // input messages (const ref)
                                       const std::string & system_prompt,
                                       rapidjson::Document::AllocatorType& allocator);
};
};
