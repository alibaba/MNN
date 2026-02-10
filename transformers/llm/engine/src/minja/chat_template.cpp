/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#ifdef LLM_USE_MINJA
#include "minja.hpp"
#include "chat_template.hpp"
namespace minja {
    // Helper to convert Value to string
    static std::string valueToString(const rapidjson::Value& val) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        val.Accept(writer);
        return buffer.GetString();
    }
    chat_template::chat_template(const std::string & source, const std::string & bos_token, const std::string & eos_token)
        : source_(source), bos_token_(bos_token), eos_token_(eos_token)
    {
        template_root_ = minja::parse(source_, {
                /* .trim_blocks = */ true,
                /* .lstrip_blocks = */ true,
                /* .keep_trailing_newline = */ false,
                });
#define MINJA_ADD_TEST
#ifdef MINJA_ADD_TEST
        auto contains = [](const std::string & haystack, const std::string & needle) {
            return haystack.find(needle) != std::string::npos;
        };

        // This entire block needs to be refactored to use rapidjson.
        // This is a significant change due to how objects and arrays are constructed.
        // I will need a Document (with its allocator) for each JSON structure.
        Document d_render_test; // Document for constructing test JSONs
        auto& alloc = d_render_test.GetAllocator();

        const std::string user_needle = "<User Needle>";
        const std::string sys_needle = "<System Needle>";

        // const json dummy_str_user_msg = {{"role", "user"}, {"content", user_needle}};
        rapidjson::Value dummy_str_user_msg(rapidjson::kObjectType);
        dummy_str_user_msg.AddMember("role", "user", alloc);
        dummy_str_user_msg.AddMember("content", rapidjson::StringRef(user_needle.c_str()), alloc);
        // const json dummy_typed_user_msg = {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", user_needle}}})}};
        rapidjson::Value dummy_typed_user_msg(rapidjson::kObjectType);
        dummy_typed_user_msg.AddMember("role", "user", alloc);
        {
            rapidjson::Value content_array(rapidjson::kArrayType);
            rapidjson::Value content_item(rapidjson::kObjectType);
            content_item.AddMember("type", "text", alloc);
            content_item.AddMember("text", rapidjson::StringRef(user_needle.c_str()), alloc);
            content_array.PushBack(content_item, alloc);
            dummy_typed_user_msg.AddMember("content", content_array, alloc);
        }
        rapidjson::Value dummy_str_user_msg_copy1;
        dummy_str_user_msg_copy1.CopyFrom(dummy_str_user_msg, alloc);

        rapidjson::Value messages_for_render1(rapidjson::kArrayType);
        messages_for_render1.PushBack(dummy_str_user_msg_copy1, alloc);
        rapidjson::Value no_tools(rapidjson::kArrayType); // Assuming empty array for no tools

        // For capability detection, polyfills are off, so copy is fine.
        rapidjson::Value messages_typed_content_test1(rapidjson::kArrayType);
        dummy_str_user_msg_copy1.CopyFrom(dummy_str_user_msg, alloc);
        messages_typed_content_test1.PushBack(dummy_str_user_msg_copy1, alloc);
        rapidjson::Value no_tools_copy1; no_tools_copy1.CopyFrom(no_tools, alloc);

        rapidjson::Value dummy_typed_user_msg_copy1;
        dummy_typed_user_msg_copy1.CopyFrom(dummy_typed_user_msg, alloc);
        rapidjson::Value messages_typed_content_test2(rapidjson::kArrayType);
        messages_typed_content_test2.PushBack(dummy_typed_user_msg_copy1, alloc);
        rapidjson::Value no_tools_copy2; no_tools_copy2.CopyFrom(no_tools, alloc);

        caps_.requires_typed_content =
            !contains(try_raw_render(messages_typed_content_test1, no_tools_copy1, false, alloc), user_needle) &&
            contains(try_raw_render(messages_typed_content_test2, no_tools_copy2, false, alloc), user_needle);

        // const auto dummy_user_msg = caps_.requires_typed_content ? dummy_typed_user_msg : dummy_str_user_msg;
        rapidjson::Value dummy_user_msg(rapidjson::kObjectType);
        if (caps_.requires_typed_content) {
            dummy_user_msg.CopyFrom(dummy_typed_user_msg, alloc);
        } else {
            dummy_user_msg.CopyFrom(dummy_str_user_msg, alloc);
        }

        // const json needle_system_msg = {
        //     {"role", "system"},
        //     {"content", caps_.requires_typed_content ? json::array({{{"type", "text"}, {"text", sys_needle}}}) : json(sys_needle)},
        // };
        rapidjson::Value needle_system_msg(rapidjson::kObjectType);
        needle_system_msg.AddMember("role", "system", alloc);
        if (caps_.requires_typed_content) {
            rapidjson::Value content_array_sys(rapidjson::kArrayType);
            rapidjson::Value content_item_sys(rapidjson::kObjectType);
            content_item_sys.AddMember("type", "text", alloc);
            content_item_sys.AddMember("text", rapidjson::StringRef(sys_needle.c_str()), alloc);
            content_array_sys.PushBack(content_item_sys, alloc);
            needle_system_msg.AddMember("content", content_array_sys, alloc);
        } else {
            needle_system_msg.AddMember("content", rapidjson::StringRef(sys_needle.c_str()), alloc);
        }

        // caps_.supports_system_role = contains(try_raw_render({needle_system_msg, dummy_user_msg,}, {}, false), sys_needle);
        rapidjson::Value messages_for_sys_role_test(rapidjson::kArrayType);
        rapidjson::Value needle_system_msg_copy; needle_system_msg_copy.CopyFrom(needle_system_msg, alloc);
        rapidjson::Value dummy_user_msg_copy2; dummy_user_msg_copy2.CopyFrom(dummy_user_msg, alloc);
        messages_for_sys_role_test.PushBack(needle_system_msg_copy, alloc);
        messages_for_sys_role_test.PushBack(dummy_user_msg_copy2, alloc);
        rapidjson::Value no_tools_copy3; no_tools_copy3.CopyFrom(no_tools, alloc);
        caps_.supports_system_role = contains(try_raw_render(messages_for_sys_role_test, no_tools_copy3, false, alloc), sys_needle);

        // auto out = try_raw_render(json::array({dummy_user_msg}), json::array({...}), false);
        rapidjson::Value messages_for_tools_test(rapidjson::kArrayType);
        rapidjson::Value dummy_user_msg_copy3; dummy_user_msg_copy3.CopyFrom(dummy_user_msg, alloc);
        messages_for_tools_test.PushBack(dummy_user_msg_copy3, alloc);

        rapidjson::Value tools_for_test(rapidjson::kArrayType);
        rapidjson::Value tool_def(rapidjson::kObjectType);
        tool_def.AddMember("name", "some_tool", alloc);
        tool_def.AddMember("type", "function", alloc);
        rapidjson::Value function_def(rapidjson::kObjectType);
        function_def.AddMember("name", "some_tool", alloc);
        function_def.AddMember("description", "Some tool.", alloc);
        rapidjson::Value params_def(rapidjson::kObjectType);
        params_def.AddMember("type", "object", alloc);
        rapidjson::Value props_def(rapidjson::kObjectType);
        rapidjson::Value arg_def(rapidjson::kObjectType);
        arg_def.AddMember("type", "string", alloc);
        arg_def.AddMember("description", "Some argument.", alloc);
        props_def.AddMember("arg", arg_def, alloc);
        params_def.AddMember("properties", props_def, alloc);
        rapidjson::Value required_arr(rapidjson::kArrayType);
        required_arr.PushBack("arg", alloc);
        params_def.AddMember("required", required_arr, alloc);
        function_def.AddMember("parameters", params_def, alloc);
        tool_def.AddMember("function", function_def, alloc);
        tools_for_test.PushBack(tool_def, alloc);

        std::string out_tools_test = try_raw_render(tools_for_test, tools_for_test, false, alloc);
        caps_.supports_tools = contains(out_tools_test, "some_tool");

        // auto make_tool_calls_msg = [&](const json & tool_calls) { ... }
        auto make_tool_calls_msg_rj = [&](rapidjson::Value& tool_calls_val, rapidjson::Document::AllocatorType& allocator_func) {
            rapidjson::Value msg(rapidjson::kObjectType);
            msg.AddMember("role", "assistant", allocator_func);
            msg.AddMember("content", rapidjson::Value(rapidjson::kNullType), allocator_func);
            msg.AddMember("tool_calls", tool_calls_val, allocator_func); // tool_calls_val is already using alloc from caller
            return msg;
        };

        // auto make_tool_call = [](const std::string & tool_name, const json & arguments) { ... }
        auto make_tool_call_rj = [&](const char* tool_name_str, rapidjson::Value& arguments_val, rapidjson::Document::AllocatorType& allocator_func) {
            rapidjson::Value tc(rapidjson::kObjectType);
            tc.AddMember("id", "call_1___", allocator_func);
            tc.AddMember("type", "function", allocator_func);
            rapidjson::Value func(rapidjson::kObjectType);
            func.AddMember("arguments", arguments_val, allocator_func); // arguments_val is already using alloc from caller
            func.AddMember("name", rapidjson::StringRef(tool_name_str), allocator_func);
            tc.AddMember("function", func, allocator_func);
            return tc;
        };

        // const json dummy_args_obj {{"argument_needle", "print('Hello, World!')"}};
        rapidjson::Value dummy_args_obj_rj(rapidjson::kObjectType);
        dummy_args_obj_rj.AddMember("argument_needle", "print('Hello, World!')", alloc);

        // Convert dummy_args_obj_rj to string for the first test
        rapidjson::StringBuffer buffer_args_str;
        rapidjson::Writer<rapidjson::StringBuffer> writer_args_str(buffer_args_str);
        dummy_args_obj_rj.Accept(writer_args_str);
        std::string dummy_args_obj_as_string = buffer_args_str.GetString();
        rapidjson::Value dummy_args_str_val(dummy_args_obj_as_string.c_str(), alloc);


        // out = try_raw_render(json::array({ dummy_user_msg, make_tool_calls_msg(json::array({make_tool_call("ipython", dummy_args_obj.dump())})) }), {}, false);
        rapidjson::Value messages_for_tool_call_str_args_test(rapidjson::kArrayType);
        rapidjson::Value dummy_user_msg_copy4; dummy_user_msg_copy4.CopyFrom(dummy_user_msg, alloc);
        messages_for_tool_call_str_args_test.PushBack(dummy_user_msg_copy4, alloc);
        rapidjson::Value tool_calls_array1(rapidjson::kArrayType);
        rapidjson::Value tc1_args_str; tc1_args_str.CopyFrom(dummy_args_str_val, alloc); // Already a string value
        std::string ipython = "ipython";
        tool_calls_array1.PushBack(make_tool_call_rj(ipython.c_str(), tc1_args_str, alloc), alloc);
        rapidjson::Value tool_calls_msg1 = make_tool_calls_msg_rj(tool_calls_array1, alloc);
        messages_for_tool_call_str_args_test.PushBack(tool_calls_msg1, alloc);
        rapidjson::Value no_tools_copy4; no_tools_copy4.CopyFrom(no_tools, alloc);
        std::string out_tool_call_str_args = try_raw_render(messages_for_tool_call_str_args_test, no_tools_copy4, false, alloc);
        bool tool_call_renders_str_arguments = contains(out_tool_call_str_args, "\"argument_needle\":") || contains(out_tool_call_str_args, "'argument_needle':");

        // out = try_raw_render(json::array({ dummy_user_msg, make_tool_calls_msg(json::array({make_tool_call("ipython", dummy_args_obj)})) }), {}, false);
        rapidjson::Value messages_for_tool_call_obj_args_test(rapidjson::kArrayType);
        rapidjson::Value dummy_user_msg_copy5; dummy_user_msg_copy5.CopyFrom(dummy_user_msg, alloc);
        messages_for_tool_call_obj_args_test.PushBack(dummy_user_msg_copy5, alloc);
        rapidjson::Value tool_calls_array2(rapidjson::kArrayType);
        rapidjson::Value tc1_args_obj; tc1_args_obj.CopyFrom(dummy_args_obj_rj, alloc);
        tool_calls_array2.PushBack(make_tool_call_rj("ipython", tc1_args_obj, alloc), alloc);
        rapidjson::Value tool_calls_msg2 = make_tool_calls_msg_rj(tool_calls_array2, alloc);
        messages_for_tool_call_obj_args_test.PushBack(tool_calls_msg2, alloc);
        rapidjson::Value no_tools_copy5; no_tools_copy5.CopyFrom(no_tools, alloc);
        std::string out_tool_call_obj_args = try_raw_render(messages_for_tool_call_obj_args_test, no_tools_copy5, false, alloc);
        bool tool_call_renders_obj_arguments = contains(out_tool_call_obj_args, "\"argument_needle\":") || contains(out_tool_call_obj_args, "'argument_needle':");

        caps_.supports_tool_calls = tool_call_renders_str_arguments || tool_call_renders_obj_arguments;
        caps_.requires_object_arguments = !tool_call_renders_str_arguments && tool_call_renders_obj_arguments;

        // auto out_empty = try_raw_render(json::array({dummy_user_msg, {{"role", "assistant"}, {"content", ""}}}), {}, false);
        rapidjson::Value messages_for_empty_content_test(rapidjson::kArrayType);
        rapidjson::Value dummy_user_msg_copy6; dummy_user_msg_copy6.CopyFrom(dummy_user_msg, alloc);
        messages_for_empty_content_test.PushBack(dummy_user_msg_copy6, alloc);
        rapidjson::Value assistant_msg_empty_content(rapidjson::kObjectType);
        assistant_msg_empty_content.AddMember("role", "assistant", alloc);
        assistant_msg_empty_content.AddMember("content", "", alloc);
        messages_for_empty_content_test.PushBack(assistant_msg_empty_content, alloc);
        rapidjson::Value no_tools_copy6; no_tools_copy6.CopyFrom(no_tools, alloc);
        std::string out_empty_content = try_raw_render(messages_for_empty_content_test, no_tools_copy6, false, alloc);

        // auto out_null = try_raw_render(json::array({dummy_user_msg, {{"role", "assistant"}, {"content", nullptr}}}), {}, false);
        rapidjson::Value messages_for_null_content_test(rapidjson::kArrayType);
        rapidjson::Value dummy_user_msg_copy7; dummy_user_msg_copy7.CopyFrom(dummy_user_msg, alloc);
        messages_for_null_content_test.PushBack(dummy_user_msg_copy7, alloc);
        rapidjson::Value assistant_msg_null_content(rapidjson::kObjectType);
        assistant_msg_null_content.AddMember("role", "assistant", alloc);
        assistant_msg_null_content.AddMember("content", rapidjson::Value(rapidjson::kNullType), alloc);
        messages_for_null_content_test.PushBack(assistant_msg_null_content, alloc);
        rapidjson::Value no_tools_copy7; no_tools_copy7.CopyFrom(no_tools, alloc);
        std::string out_null_content = try_raw_render(messages_for_null_content_test, no_tools_copy7, false, alloc);
        caps_.requires_non_null_content = contains(out_empty_content, user_needle) && !contains(out_null_content, user_needle);


        if (caps_.supports_tool_calls) {
            // auto dummy_args = caps_.requires_object_arguments ? dummy_args_obj : json(dummy_args_obj.dump());
            rapidjson::Value dummy_args_for_parallel_test;
            if (caps_.requires_object_arguments) {
                dummy_args_for_parallel_test.CopyFrom(dummy_args_obj_rj, alloc);
            } else {
                // This was already created: dummy_args_str_val (string version of dummy_args_obj_rj)
                dummy_args_for_parallel_test.CopyFrom(dummy_args_str_val, alloc);
            }

            // auto tc1 = make_tool_call("test_tool1", dummy_args);
            // auto tc2 = make_tool_call("test_tool2", dummy_args);
            rapidjson::Value dummy_args_tc1; dummy_args_tc1.CopyFrom(dummy_args_for_parallel_test, alloc);
            rapidjson::Value tc1 = make_tool_call_rj("test_tool1", dummy_args_tc1, alloc);
            rapidjson::Value dummy_args_tc2; dummy_args_tc2.CopyFrom(dummy_args_for_parallel_test, alloc);
            rapidjson::Value tc2 = make_tool_call_rj("test_tool2", dummy_args_tc2, alloc);

            // auto out = try_raw_render(json::array({ dummy_user_msg, make_tool_calls_msg(json::array({tc1, tc2})) }), {}, false);
            rapidjson::Value messages_for_parallel_calls_test(rapidjson::kArrayType);
            rapidjson::Value dummy_user_msg_copy8; dummy_user_msg_copy8.CopyFrom(dummy_user_msg, alloc);
            messages_for_parallel_calls_test.PushBack(dummy_user_msg_copy8, alloc);
            rapidjson::Value tool_calls_array_parallel(rapidjson::kArrayType);
            tool_calls_array_parallel.PushBack(tc1, alloc); // tc1, tc2 are already using alloc
            tool_calls_array_parallel.PushBack(tc2, alloc);
            rapidjson::Value tool_calls_msg_parallel = make_tool_calls_msg_rj(tool_calls_array_parallel, alloc);
            messages_for_parallel_calls_test.PushBack(tool_calls_msg_parallel, alloc);
            rapidjson::Value no_tools_copy8; no_tools_copy8.CopyFrom(no_tools, alloc);
            std::string out_parallel_calls = try_raw_render(messages_for_parallel_calls_test, no_tools_copy8, false, alloc);
            caps_.supports_parallel_tool_calls = contains(out_parallel_calls, "test_tool1") && contains(out_parallel_calls, "test_tool2");

            // Need to re-create tc1 as it was moved into tool_calls_array_parallel
            rapidjson::Value dummy_args_tc1_resp; dummy_args_tc1_resp.CopyFrom(dummy_args_for_parallel_test, alloc);
            rapidjson::Value tc1_resp = make_tool_call_rj("test_tool1", dummy_args_tc1_resp, alloc);

            // out = try_raw_render(json::array({ dummy_user_msg, make_tool_calls_msg(json::array({tc1})), { ...tool response... } }), {}, false);
            rapidjson::Value messages_for_tool_response_test(rapidjson::kArrayType);
            rapidjson::Value dummy_user_msg_copy9; dummy_user_msg_copy9.CopyFrom(dummy_user_msg, alloc);
            messages_for_tool_response_test.PushBack(dummy_user_msg_copy9, alloc);
            rapidjson::Value tool_calls_array_resp(rapidjson::kArrayType);
            tool_calls_array_resp.PushBack(tc1_resp, alloc);
            rapidjson::Value tool_calls_msg_resp = make_tool_calls_msg_rj(tool_calls_array_resp, alloc);
            messages_for_tool_response_test.PushBack(tool_calls_msg_resp, alloc);
            rapidjson::Value tool_response_msg(rapidjson::kObjectType);
            tool_response_msg.AddMember("role", "tool", alloc);
            tool_response_msg.AddMember("name", "test_tool1", alloc);
            tool_response_msg.AddMember("content", "Some response!", alloc);
            tool_response_msg.AddMember("tool_call_id", "call_911_", alloc);
            messages_for_tool_response_test.PushBack(tool_response_msg, alloc);
            rapidjson::Value no_tools_copy9; no_tools_copy9.CopyFrom(no_tools, alloc);
            std::string out_tool_response = try_raw_render(messages_for_tool_response_test, no_tools_copy9, false, alloc);
            caps_.supports_tool_responses = contains(out_tool_response, "Some response!");
            caps_.supports_tool_call_id = contains(out_tool_response, "call_911_");
        }

        if (!caps_.supports_tools) {
            // const json user_msg { {"role", "user"}, {"content", "Hey"} };
            rapidjson::Value user_msg_infer(rapidjson::kObjectType);
            user_msg_infer.AddMember("role", "user", alloc);
            user_msg_infer.AddMember("content", "Hey", alloc);

            // const json args { {"arg1", "some_value"} };
            rapidjson::Value args_infer(rapidjson::kObjectType);
            args_infer.AddMember("arg1", "some_value", alloc);

            // const json tool_call_msg { ... }
            rapidjson::Value tool_call_msg_infer(rapidjson::kObjectType);
            tool_call_msg_infer.AddMember("role", "assistant", alloc);
            tool_call_msg_infer.AddMember("content", rapidjson::Value(rapidjson::kNullType), alloc);
            rapidjson::Value tool_calls_array_infer(rapidjson::kArrayType);
            rapidjson::Value tool_call_item_infer(rapidjson::kObjectType);
            tool_call_item_infer.AddMember("id", "call_1___", alloc);
            tool_call_item_infer.AddMember("type", "function", alloc);
            rapidjson::Value function_item_infer(rapidjson::kObjectType);
            function_item_infer.AddMember("name", "tool_name", alloc);

            rapidjson::Value arguments_infer;
            if (caps_.requires_object_arguments) {
                arguments_infer.CopyFrom(args_infer, alloc);
            } else {
                // This requires minja::Value::dump which itself uses nlohmann::json.
                // This part needs a temporary nlohmann::json to dump, or reimplement dump logic for rapidjson.
                // For now, let's assume minja::Value can give us a string that rapidjson can parse,
                // or we construct the string directly.
                // minja::Value(args).dump(-1, /* to_json= */ true)
                // This is a major dependency. For now, I'll create a simple string version.
                rapidjson::StringBuffer buffer_args_infer_str;
                rapidjson::Writer<rapidjson::StringBuffer> writer_args_infer_str(buffer_args_infer_str);
                args_infer.Accept(writer_args_infer_str);
                arguments_infer.SetString(buffer_args_infer_str.GetString(), alloc);
            }
            function_item_infer.AddMember("arguments", arguments_infer, alloc);
            tool_call_item_infer.AddMember("function", function_item_infer, alloc);
            tool_calls_array_infer.PushBack(tool_call_item_infer, alloc);
            tool_call_msg_infer.AddMember("tool_calls", tool_calls_array_infer, alloc);

            std::string prefix_str, full_str;
            {
                chat_template_inputs inputs_prefix;
                inputs_prefix.allocator_for_inputs = &alloc;
                inputs_prefix.messages.SetArray();
                rapidjson::Value user_msg_infer_copy1; user_msg_infer_copy1.CopyFrom(user_msg_infer, alloc);
                inputs_prefix.messages.PushBack(user_msg_infer_copy1, alloc);
                inputs_prefix.add_generation_prompt = true;
                // inputs.tools is already kNullType by default in chat_template_inputs constructor
                prefix_str = apply(inputs_prefix);
            }
            {
                chat_template_inputs inputs_full;
                inputs_full.allocator_for_inputs = &alloc;
                inputs_full.messages.SetArray();
                rapidjson::Value user_msg_infer_copy2; user_msg_infer_copy2.CopyFrom(user_msg_infer, alloc);
                inputs_full.messages.PushBack(user_msg_infer_copy2, alloc);
                rapidjson::Value tool_call_msg_infer_copy; tool_call_msg_infer_copy.CopyFrom(tool_call_msg_infer, alloc);
                inputs_full.messages.PushBack(tool_call_msg_infer_copy, alloc);
                inputs_full.add_generation_prompt = false;
                // inputs.tools is already kNullType by default
                full_str = apply(inputs_full);
            }
            // ... rest of the logic for tool_call_example_ using prefix_str and full_str
            // This part seems okay to remain as string manipulation
            auto eos_pos_last = full_str.rfind(eos_token_);
            if (eos_pos_last == prefix_str.size() - eos_token_.size() ||
                    (full_str[full_str.size() - 1] == '\n' && (eos_pos_last == full_str.size() - eos_token_.size() - 1))) {
                full_str = full_str.substr(0, eos_pos_last);
            }
            size_t common_prefix_length = 0;
            for (size_t i = 0; i < prefix_str.size() && i < full_str.size(); ++i) {
                if (prefix_str[i] != full_str[i]) {
                    break;
                }
                if (prefix_str[i] == '<') {
                    continue;
                }
                common_prefix_length = i + 1;
            }
            auto example = full_str.substr(common_prefix_length);
            if (example.find("tool_name") == std::string::npos && example.find("some_value") == std::string::npos) {
                fprintf(stderr, "Failed to infer a tool call example (possible template bug)\n");
            } else {
                tool_call_example_ = example;
            }
        }
        // Ensure d_render_test is cleared if it were a member, but it's local.
#endif
    }
    std::string chat_template::try_raw_render(
            rapidjson::Value& messages, // Modifying to pass by ref as it might be changed by polyfills later
            rapidjson::Value& tools,    // Modifying to pass by ref
            bool add_generation_prompt,
            rapidjson::Document::AllocatorType& allocator, // Added allocator
            rapidjson::Value extra_context) const // Default to null
    {
        chat_template_inputs inputs;
        // Important: When assigning Value, if it's from another Document or a temporary,
        // it needs to be deep copied using the allocator of the target Document/Value.
        // For try_raw_render, we assume messages, tools, extra_context are already managed
        // or will be properly constructed with an allocator.
        // Here, we're creating new Value objects for the inputs struct, so they need an allocator
        // if they are to be populated. However, inputs here is temporary.
        // The original nlohmann version copied, rapidjson Value assignment is a shallow copy.
        // This needs careful handling. For now, let's assume the caller manages lifetime.
        // This is tricky because the Value objects in chat_template_inputs need an allocator.
        // Let's try to pass the allocator to inputs.
        inputs.allocator_for_inputs = &allocator;
        inputs.messages.CopyFrom(messages, allocator);
        inputs.tools.CopyFrom(tools, allocator);
        inputs.add_generation_prompt = add_generation_prompt;
        if (!extra_context.IsNull()) {
            inputs.extra_context.CopyFrom(extra_context, allocator);
        } else {
            inputs.extra_context.SetObject(); // Initialize as empty object if default
        }
        // Use fixed date for tests
        inputs.now = std::chrono::system_clock::from_time_t(0);

        chat_template_options opts;
        opts.apply_polyfills = false;

        auto prompt = apply(inputs, opts);
        // fprintf(stderr, "try_raw_render: %s\n", prompt.c_str());
        return prompt;
    }
    std::string chat_template::apply(
            chat_template_inputs & inputs,
            const chat_template_options & opts) const {
                AUTOTIME;
        // Create a working document for this apply call.
        // All new JSON Values created within this scope should use its allocator.
        Document working_doc;
        rapidjson::Document::AllocatorType& allocator = working_doc.GetAllocator();

        rapidjson::Value actual_messages(rapidjson::kArrayType); // Uses working_doc's allocator by default if created here

        auto has_tools = inputs.tools.IsArray() && !inputs.tools.Empty();
        auto has_tool_calls = false;
        auto has_tool_responses = false;
        auto has_string_content = false;

        if (inputs.messages.IsArray()) {
            for (const auto & message_val : inputs.messages.GetArray()) {
                if (message_val.IsObject()) {
                    if (message_val.HasMember("tool_calls") && !message_val["tool_calls"].IsNull()) {
                        has_tool_calls = true;
                    }
                    if (message_val.HasMember("role") && message_val["role"].IsString() &&
                            strcmp(message_val["role"].GetString(), "tool") == 0) {
                        has_tool_responses = true;
                    }
                    if (message_val.HasMember("content") && message_val["content"].IsString()) {
                        has_string_content = true;
                    }
                }
            }
        }

        auto polyfill_system_role = opts.polyfill_system_role && !caps_.supports_system_role;
        auto polyfill_tools = opts.polyfill_tools && has_tools && !caps_.supports_tools;
        auto polyfill_tool_call_example = polyfill_tools && opts.polyfill_tool_call_examples;
        auto polyfill_tool_calls = opts.polyfill_tool_calls && has_tool_calls && !caps_.supports_tool_calls;
        auto polyfill_tool_responses = opts.polyfill_tool_responses && has_tool_responses && !caps_.supports_tool_responses;
        auto polyfill_object_arguments = opts.polyfill_object_arguments && has_tool_calls && caps_.requires_object_arguments;
        auto polyfill_typed_content = opts.polyfill_typed_content && has_string_content && caps_.requires_typed_content;

        auto needs_polyfills = opts.apply_polyfills && (false
                || polyfill_system_role
                || polyfill_tools
                || polyfill_tool_calls
                || polyfill_tool_responses
                || polyfill_object_arguments
                || polyfill_typed_content
                );

        if (needs_polyfills) {
            // actual_messages is already an empty array, using allocator

            auto add_message = [&](const rapidjson::Value & msg_const) {
                rapidjson::Value msg;
                msg.CopyFrom(msg_const, allocator); // Ensure it uses the current doc's allocator

                if (polyfill_typed_content && msg.IsObject() && msg.HasMember("content") &&
                        !msg["content"].IsNull() && msg["content"].IsString()) {

                    rapidjson::Value new_msg(rapidjson::kObjectType);
                    new_msg.AddMember("role", rapidjson::Value(msg["role"], allocator), allocator); // copy role

                    rapidjson::Value content_array_typed(rapidjson::kArrayType);
                    rapidjson::Value content_item_typed(rapidjson::kObjectType);
                    content_item_typed.AddMember("type", "text", allocator);
                    // Need to copy the string content for "text"
                    rapidjson::Value text_val(msg["content"].GetString(), allocator);
                    content_item_typed.AddMember("text", text_val, allocator);
                    content_array_typed.PushBack(content_item_typed, allocator);
                    new_msg.AddMember("content", content_array_typed, allocator);
                    actual_messages.PushBack(new_msg, allocator);
                } else {
                    actual_messages.PushBack(msg, allocator); // msg already copied with allocator
                }
            };

            std::string pending_system;
            auto flush_sys = [&]() {
                if (!pending_system.empty()) {
                    rapidjson::Value sys_as_user_msg(rapidjson::kObjectType);
                    sys_as_user_msg.AddMember("role", "user", allocator);
                    sys_as_user_msg.AddMember("content", rapidjson::StringRef(pending_system.c_str()), allocator);
                    add_message(sys_as_user_msg); // add_message will handle typed content if needed
                    pending_system.clear();
                }
            };

            rapidjson::Value adjusted_messages_val(rapidjson::kArrayType);
            if (polyfill_tools) {
                // Convert inputs.tools to string for the system prompt
                rapidjson::StringBuffer tools_buffer;
                rapidjson::PrettyWriter<rapidjson::StringBuffer> tools_writer(tools_buffer); // Pretty for readability
                tools_writer.SetIndent(' ', 2);
                inputs.tools.Accept(tools_writer);
                std::string tools_str_prompt = tools_buffer.GetString();

                std::string system_prompt_str =
                    "You can call any of the following tools to satisfy the user's requests: " + tools_str_prompt +
                    (!polyfill_tool_call_example || tool_call_example_.empty() ? "" : "\n\nExample tool call syntax:\n\n" + tool_call_example_ + "\n\n");

                // add_system returns a new Value, ensure it uses 'allocator'
                rapidjson::Value messages_copy_for_add_system;
                messages_copy_for_add_system.CopyFrom(inputs.messages, allocator);
                adjusted_messages_val = add_system(messages_copy_for_add_system, system_prompt_str, allocator);
            } else {
                adjusted_messages_val.CopyFrom(inputs.messages, allocator);
            }

            if (adjusted_messages_val.IsArray()){
                for (auto & message_val_mut : adjusted_messages_val.GetArray()) { // Iterate by mutable ref
                                                                                  // message_ is already using 'allocator' as it's part of adjusted_messages_val
                    rapidjson::Value message; // Create a mutable copy for this iteration
                    message.CopyFrom(message_val_mut, allocator);


                    if (!message.IsObject() || !message.HasMember("role") || !message.HasMember("content")) {
                        // MNN_ERROR replacement:
                        fprintf(stderr, "message must have 'role' and 'content' fields: %s\n", valueToString(message).c_str());
                        // Potentially skip this message or handle error
                        continue;
                    }
                    const char* role_cstr = message["role"].GetString();
                    std::string role = role_cstr;

                    if (message.HasMember("tool_calls")) {
                        if (polyfill_object_arguments || polyfill_tool_calls) {
                            if (message["tool_calls"].IsArray()) {
                                for (auto & tool_call_val : message["tool_calls"].GetArray()) {
                                    if (tool_call_val.IsObject() && tool_call_val.HasMember("type") && tool_call_val["type"] == "function") {
                                        if (tool_call_val.HasMember("function") && tool_call_val["function"].IsObject()) {
                                            auto& function_val = tool_call_val["function"];
                                            if (function_val.HasMember("arguments") && function_val["arguments"].IsString()) {
                                                std::string args_str = function_val["arguments"].GetString();
                                                Document args_doc;
                                                if (!args_doc.Parse(args_str.c_str()).HasParseError()) {
                                                    // Replace the string arguments with the parsed Value object
                                                    // The new Value must use 'allocator'
                                                    rapidjson::Value new_args_val;
                                                    new_args_val.CopyFrom(args_doc, allocator);
                                                    function_val["arguments"].Swap(new_args_val); // Swap to avoid copy if possible
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (polyfill_tool_calls) {
                            rapidjson::Value content_val; content_val.CopyFrom(message["content"], allocator); // Keep original content if any
                            rapidjson::Value tool_calls_payload(rapidjson::kArrayType);
                            if (message["tool_calls"].IsArray()) {
                                for (const auto & tool_call_val_const : message["tool_calls"].GetArray()) {
                                    if (tool_call_val_const.IsObject() && tool_call_val_const.HasMember("type") && tool_call_val_const["type"] == "function") {
                                        const auto& function_val_const = tool_call_val_const["function"];
                                        rapidjson::Value tc_item(rapidjson::kObjectType);
                                        tc_item.AddMember("name", rapidjson::Value(function_val_const["name"], allocator), allocator);
                                        // Arguments should already be objects if polyfill_object_arguments ran
                                        tc_item.AddMember("arguments", rapidjson::Value(function_val_const["arguments"], allocator), allocator);
                                        if (tool_call_val_const.HasMember("id")) {
                                            tc_item.AddMember("id", rapidjson::Value(tool_call_val_const["id"], allocator), allocator);
                                        }
                                        tool_calls_payload.PushBack(tc_item, allocator);
                                    }
                                }
                            }
                            rapidjson::Value obj_for_content(rapidjson::kObjectType);
                            obj_for_content.AddMember("tool_calls", tool_calls_payload, allocator);
                            if (!content_val.IsNull() && !(content_val.IsString() && strlen(content_val.GetString()) == 0)) {
                                obj_for_content.AddMember("content", content_val, allocator);
                            }

                            // Serialize obj_for_content to string for message["content"]
                            rapidjson::StringBuffer s_buffer;
                            rapidjson::PrettyWriter<rapidjson::StringBuffer> writer_obj(s_buffer);
                            writer_obj.SetIndent(' ', 2);
                            obj_for_content.Accept(writer_obj);
                            message["content"].SetString(s_buffer.GetString(), allocator);
                            message.RemoveMember("tool_calls");
                        }
                    }
                    if (polyfill_tool_responses && role == "tool") {
                        message["role"].SetString("user", allocator); // Change role to user
                        rapidjson::Value tool_response_obj(rapidjson::kObjectType);
                        rapidjson::Value tool_response_inner_obj(rapidjson::kObjectType);

                        if (message.HasMember("name")) {
                            tool_response_inner_obj.AddMember("tool", rapidjson::Value(message["name"], allocator), allocator);
                        }
                        // message["content"] is guaranteed to exist by check above
                        tool_response_inner_obj.AddMember("content", rapidjson::Value(message["content"], allocator), allocator);
                        if (message.HasMember("tool_call_id")) {
                            tool_response_inner_obj.AddMember("tool_call_id", rapidjson::Value(message["tool_call_id"], allocator), allocator);
                        }
                        tool_response_obj.AddMember("tool_response", tool_response_inner_obj, allocator);

                        // Serialize tool_response_obj to string for message["content"]
                        rapidjson::StringBuffer s_buffer_resp;
                        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer_resp(s_buffer_resp);
                        writer_resp.SetIndent(' ',2);
                        tool_response_obj.Accept(writer_resp);
                        message["content"].SetString(s_buffer_resp.GetString(), allocator);

                        if (message.HasMember("name")) message.RemoveMember("name");
                        if (message.HasMember("tool_call_id")) message.RemoveMember("tool_call_id"); // if it was there
                    }

                    if (!message["content"].IsNull() && polyfill_system_role) {
                        // Assuming content is string after previous polyfills or by its nature
                        std::string content_str;
                        if (message["content"].IsString()){
                            content_str = message["content"].GetString();
                        } else {
                            // If content is not string (e.g. array for typed content), it needs to be stringified for pending_system
                            // This case should be handled by typed_content polyfill first if active
                            // For simplicity, if it's not string here, we might skip or stringify it
                            rapidjson::StringBuffer temp_s_buffer;
                            rapidjson::Writer<rapidjson::StringBuffer> temp_writer(temp_s_buffer);
                            message["content"].Accept(temp_writer);
                            content_str = temp_s_buffer.GetString();
                        }

                        if (role == "system") {
                            if (!pending_system.empty()) pending_system += "\n";
                            pending_system += content_str;
                            // This message is consumed, skip adding it directly
                            // A continue here would skip the 'add_message(message)' below for system messages
                            // which is the desired behavior.
                            // However, the original code structure adds the modified message (if not system)
                            // or flushes system messages.
                            // Let's ensure this message isn't added by 'add_message' if it's system.
                            // The flush_sys() and add_message(message) logic outside the loop handles it.
                            // So, if role is system, we just update pending_system and the message itself is not added.
                            continue;
                        } else {
                            if (role == "user") {
                                if (!pending_system.empty()) {
                                    std::string new_content = pending_system + (content_str.empty() ? "" : "\n" + content_str);
                                    message["content"].SetString(new_content.c_str(), allocator);
                                    pending_system.clear();
                                }
                            } else { // assistant, tool (already transformed to user)
                                flush_sys();
                            }
                        }
                    }
                    add_message(message); // add_message handles copying to actual_messages with allocator
                }
            }
            flush_sys();
        } else { // no polyfills needed
            actual_messages.CopyFrom(inputs.messages, allocator);
        }

        auto context = minja::Context::make(nullptr); // nlohmann::json() equivalent for context data
                                                      // The make function needs to be adapted for rapidjson::Value
                                                      // For now, creating an empty object for context data.
        rapidjson::Value context_data_val(rapidjson::kObjectType);
        context_data_val.AddMember("messages", actual_messages, allocator); // actual_messages already uses allocator
        context_data_val.AddMember("add_generation_prompt", inputs.add_generation_prompt, allocator);


        // Convert context_data_val to nlohmann::json for minja::Context::make
        // This is a temporary bridge. minja::Context itself needs to be updated for rapidjson.
        // This is a critical dependency.

        context = minja::Context::make(minja::Value(context_data_val));

        context->set("bos_token", opts.use_bos_token ? bos_token_ : "");
        context->set("eos_token", opts.use_eos_token ? eos_token_ : "");
        if (opts.define_strftime_now) {
            auto time_now_capture = inputs.now; // capture for lambda
            context->set("strftime_now", minja::Value::callable([time_now_capture](const std::shared_ptr<minja::Context> &, minja::ArgumentsValue & args) {
                        args.expectArgs("strftime_now", {1, 1}, {0, 0});
                        auto format = args.args[0].get<std::string>();

                        auto time_point = std::chrono::system_clock::to_time_t(time_now_capture);
                        auto local_time = *std::localtime(&time_point);
                        std::ostringstream ss;
                        ss << std::put_time(&local_time, format.c_str());
                        return ss.str();
                        }));
        }

        if (!inputs.tools.IsNull()) {
            context->set("tools", minja::Value(inputs.tools));
        }
        if (!inputs.extra_context.IsNull() && inputs.extra_context.IsObject()) {
            for (auto & kv : inputs.extra_context.GetObject()) {
                context->set(kv.name.GetString(), minja::Value(kv.value));
            }
        }

        auto ret = template_root_->render(context);
        return ret;
    }
    rapidjson::Value chat_template::add_system(
            const rapidjson::Value & messages_const, // input messages (const ref)
            const std::string & system_prompt,
            rapidjson::Document::AllocatorType& allocator) {
        rapidjson::Value messages_with_system(rapidjson::kArrayType);
        messages_with_system.CopyFrom(messages_const, allocator); // Deep copy to make it modifiable

        if (!messages_with_system.Empty() && messages_with_system[0].IsObject() &&
                messages_with_system[0].HasMember("role") && messages_with_system[0]["role"] == "system") {

            std::string existing_system_content_str;
            if (messages_with_system[0].HasMember("content") && messages_with_system[0]["content"].IsString()) {
                existing_system_content_str = messages_with_system[0]["content"].GetString();
            }

            std::string new_content_str = existing_system_content_str + "\n\n" + system_prompt;
            messages_with_system[0]["content"].SetString(new_content_str.c_str(), allocator);

        } else {
            rapidjson::Value new_system_msg(rapidjson::kObjectType);
            new_system_msg.AddMember("role", "system", allocator);
            new_system_msg.AddMember("content", rapidjson::StringRef(system_prompt.c_str()), allocator);

            // Insert at the beginning
            rapidjson::Value temp_array(rapidjson::kArrayType);
            temp_array.PushBack(new_system_msg, allocator);
            for (auto& el : messages_with_system.GetArray()) {
                rapidjson::Value el_copy;
                el_copy.CopyFrom(el, allocator);
                temp_array.PushBack(el_copy, allocator);
            }
            messages_with_system.Swap(temp_array);
        }
        return messages_with_system; // This Value is allocated with 'allocator'
    }
};
#endif
