//
//  VisionTests.swift
//  Jinja
//
//  Created by Anthony DePasquale on 31.12.2024.
//

import XCTest
import OrderedCollections

@testable import Jinja

final class VisionTests: XCTestCase {
    let llama3_2visionChatTemplate =
        "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- Find out if there are any images #}\n{% set image_ns = namespace(has_images=false) %}      \n{%- for message in messages %}\n    {%- for content in message['content'] %}\n        {%- if content['type'] == 'image' %}\n            {%- set image_ns.has_images = true %}\n        {%- endif %}\n    {%- endfor %}\n{%- endfor %}\n\n{#- Error out if there are images and system message #}\n{%- if image_ns.has_images and not system_message == \"\" %}\n    {{- raise_exception(\"Prompting with images is incompatible with system messages.\") }}\n{%- endif %}\n\n{#- System message if there are no images #}\n{%- if not image_ns.has_images %}\n    {{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n    {%- if tools is not none %}\n        {{- \"Environment: ipython\\n\" }}\n    {%- endif %}\n    {{- \"Cutting Knowledge Date: December 2023\\n\" }}\n    {{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n        {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n        {{- \"Do not use variables.\\n\\n\" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- \"\\n\\n\" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- \"<|eot_id|>\" }}\n{%- endif %}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}\n        {%- if message['content'] is string %}\n            {{- message['content'] }}\n        {%- else %}\n            {%- for content in message['content'] %}\n                {%- if content['type'] == 'image' %}\n                    {{- '<|image|>' }}\n                {%- elif content['type'] == 'text' %}\n                    {{- content['text'] }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
    let qwen2VLChatTemplate =
        "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    func testLlama3_2_11BVisionInstructTextChatOnly() throws {
        let template = try Template(llama3_2visionChatTemplate)
        let result = try template.render([
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "Hello, how are you?",
                        ] as [String: Any]
                    ] as [[String: Any]],
                ] as [String: Any],
                [
                    "role": "assistant",
                    "content": [
                        [
                            "type": "text",
                            "text": "I'm doing great. How can I help you today?",
                        ] as [String: Any]
                    ] as [[String: Any]],
                ] as [String: Any],
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "I'd like to show off how chat templating works!",
                        ] as [String: Any]
                    ] as [[String: Any]],
                ] as [String: Any],
            ] as [[String: Any]] as Any,
            "bos_token": "<s>" as Any,
            "date_string": "26 Jul 2024" as Any,
            "tools_in_user_message": true as Any,
            "system_message": "You are a helpful assistant." as Any,
            "add_generation_prompt": true as Any,
        ])
        let target =
            "<s>\n<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'm doing great. How can I help you today?<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI'd like to show off how chat templating works!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        XCTAssertEqual(result, target)
    }

    func testLlama3_2_11BVisionInstructWithImages() throws {
        let template = try Template(llama3_2visionChatTemplate)
        let result = try template.render([
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "What's in this image?",
                        ] as [String: Any],
                        [
                            "type": "image",
                            "image": "base64_encoded_image_data",
                        ] as [String: Any],
                    ] as [[String: Any]],
                ] as [String: Any]
            ] as [[String: Any]],
            "bos_token": "<s>" as Any,
            "add_generation_prompt": true as Any,
        ])
        let target = """
            <s>
            <|start_header_id|>user<|end_header_id|>

            What's in this image?<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>


            """
        XCTAssertEqual(result, target)
    }

    func testQwen2VLWithImages() throws {
        let template = try Template(qwen2VLChatTemplate)
        let result = try template.render([
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "What's in this image?",
                        ] as [String: String],
                        [
                            "type": "image",
                            "image_url": "example.jpg",
                        ] as [String: String],
                    ] as [[String: String]],
                ] as [String: Any]
            ] as [[String: Any]],
            "add_generation_prompt": true,
            "add_vision_id": true,
        ])
        let target = """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            What's in this image?Picture 1: <|vision_start|><|image_pad|><|vision_end|><|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testQwen2VLWithVideo() throws {
        let template = try Template(qwen2VLChatTemplate)
        let result = try template.render([
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "What's happening in this video?",
                        ] as [String: String],
                        [
                            "type": "video",
                            "video_url": "example.mp4",
                        ] as [String: String],
                    ] as [[String: String]],
                ] as [String: Any]
            ] as [[String: Any]],
            "add_generation_prompt": true,
            "add_vision_id": true,
        ])
        let target = """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            What's happening in this video?Video 1: <|vision_start|><|video_pad|><|vision_end|><|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testLlama3_2_11BVisionInstructWithTools() throws {
        let template = try Template(llama3_2visionChatTemplate)

        let tools: [OrderedDictionary<String, Any>] = [
            OrderedDictionary(uniqueKeysWithValues: [
                ("type", "function" as Any),
                (
                    "function",
                    OrderedDictionary(uniqueKeysWithValues: [
                        ("name", "get_current_weather" as Any),
                        ("description", "Get the current weather in a given location" as Any),
                        (
                            "parameters",
                            OrderedDictionary(uniqueKeysWithValues: [
                                ("type", "object" as Any),
                                (
                                    "properties",
                                    OrderedDictionary(uniqueKeysWithValues: [
                                        (
                                            "location",
                                            OrderedDictionary(uniqueKeysWithValues: [
                                                ("type", "string" as Any),
                                                ("description", "The city and state, e.g. San Francisco, CA" as Any),
                                            ]) as Any
                                        ),
                                        (
                                            "unit",
                                            OrderedDictionary(uniqueKeysWithValues: [
                                                ("type", "string" as Any),
                                                ("enum", ["celsius", "fahrenheit"] as Any),
                                            ]) as Any
                                        ),
                                    ]) as Any
                                ),
                                ("required", ["location"] as Any),
                            ]) as Any
                        ),
                    ]) as Any
                ),
            ])
        ]

        let result = try template.render([
            "messages": [
                [
                    "role": "system",
                    "content": "You are a helpful assistant.",
                ],
                [
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                ] as [String: Any],
            ] as [[String: Any]] as Any,
            "bos_token": "<s>" as Any,
            "add_generation_prompt": true as Any,
            "tools": tools as Any,
            "tools_in_user_message": true as Any,
        ])
        let target = """
            <s>
            <|start_header_id|>system<|end_header_id|>

            Environment: ipython
            Cutting Knowledge Date: December 2023
            Today Date: \(Environment.formatDate(Date(), withFormat: "%d %b %Y"))

            You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

            Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

            Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": [
                                    "celsius",
                                    "fahrenheit"
                                ]
                            }
                        },
                        "required": [
                            "location"
                        ]
                    }
                }
            }

            What's the weather like in San Francisco?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


            """
        XCTAssertEqual(result, target)
    }
}
