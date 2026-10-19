//
//  VisionTests.swift
//  Jinja
//
//  Created by Anthony DePasquale on 30.12.2024.
//

import XCTest
import OrderedCollections

/*
 Recent models that don't support tool use:
 - Gemma 2
 - Phi 3.5
 - Mistral NeMo
 */

@testable import Jinja

final class ToolUseTests: XCTestCase {
    let messagesWithFunctionCalling: [[String: Any?]] = [
        [
            "role": "assistant",
            "content": nil,
            "tool_calls": [
                [
                    "type": "function",
                    "function": [
                        "name": "get_current_weather",
                        "arguments": "{\n  \"location\": \"Hanoi\"\n}",
                    ],
                ]
            ],
        ],
        [
            "role": "user",
            "content": "What's the weather like in Hanoi?",
        ],
    ]

    // Example adapted from https://huggingface.co/fireworks-ai/firefunction-v1
    let exampleFunctionSpec: [OrderedDictionary<String, Any>] = [
        OrderedDictionary(uniqueKeysWithValues: [
            ("name", "get_stock_price") as (String, Any),
            ("description", "Get the current stock price") as (String, Any),
            (
                "parameters",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "object") as (String, Any),
                    (
                        "properties",
                        OrderedDictionary(uniqueKeysWithValues: [
                            (
                                "symbol",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "string") as (String, Any),
                                    ("description", "The stock symbol, e.g. AAPL, GOOG") as (String, Any),
                                ])
                            )
                        ])
                    ) as (String, Any),
                    ("required", ["symbol"]) as (String, Any),
                ])
            ) as (String, Any),
        ]),
        OrderedDictionary(uniqueKeysWithValues: [
            ("name", "check_word_anagram") as (String, Any),
            ("description", "Check if two words are anagrams of each other") as (String, Any),
            (
                "parameters",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "object") as (String, Any),
                    (
                        "properties",
                        OrderedDictionary(uniqueKeysWithValues: [
                            (
                                "word1",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "string") as (String, Any),
                                    ("description", "The first word") as (String, Any),
                                ])
                            ) as (String, Any),
                            (
                                "word2",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "string") as (String, Any),
                                    ("description", "The second word") as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                    ("required", ["word1", "word2"]) as (String, Any),
                ])
            ) as (String, Any),
        ]),
    ]

    lazy var messagesWithFunctionCallingAndSystemPrompt: [OrderedDictionary<String, Any>] = [
        OrderedDictionary(uniqueKeysWithValues: [
            ("role", "system") as (String, Any),
            ("content", "You are a helpful assistant with access to functions. Use them if required.") as (String, Any),
        ]),
        OrderedDictionary(uniqueKeysWithValues: [
            ("role", "functions") as (String, Any),
            ("content", exampleFunctionSpec) as (String, Any),
        ]),
        OrderedDictionary(uniqueKeysWithValues: [
            ("role", "user") as (String, Any),
            ("content", "Hi, can you tell me the current stock price of AAPL?") as (String, Any),
        ]),
    ]

    let exampleToolJSONSchemas: OrderedDictionary<String, OrderedDictionary<String, Any>> = OrderedDictionary(
        uniqueKeysWithValues: [
            (
                "get_current_weather",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "function") as (String, Any),
                    (
                        "function",
                        OrderedDictionary(uniqueKeysWithValues: [
                            ("name", "get_current_weather") as (String, Any),
                            ("description", "Get the current weather in a given location") as (String, Any),
                            (
                                "parameters",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "properties",
                                        OrderedDictionary(uniqueKeysWithValues: [
                                            (
                                                "location",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    ("description", "The city and state, e.g. San Francisco, CA")
                                                        as (String, Any),
                                                ])
                                            ) as (String, Any),
                                            (
                                                "unit",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    ("enum", ["celsius", "fahrenheit"]) as (String, Any),
                                                ])
                                            ) as (String, Any),
                                        ])
                                    ) as (String, Any),
                                    ("required", ["location"]) as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                ])
            ),
            (
                "get_current_temperature_v1",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "function") as (String, Any),
                    (
                        "function",
                        OrderedDictionary(uniqueKeysWithValues: [
                            ("name", "get_current_temperature") as (String, Any),
                            ("description", "Get the current temperature at a location.") as (String, Any),
                            (
                                "parameters",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "properties",
                                        OrderedDictionary(uniqueKeysWithValues: [
                                            (
                                                "location",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    (
                                                        "description",
                                                        "The location to get the temperature for, in the format \"City, Country\""
                                                    ) as (String, Any),
                                                ])
                                            ) as (String, Any)
                                        ])
                                    ) as (String, Any),
                                    ("required", ["location"]) as (String, Any),
                                ])
                            ) as (String, Any),
                            (
                                "return",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "number") as (String, Any),
                                    (
                                        "description",
                                        "The current temperature at the specified location in the specified units, as a float."
                                    ) as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                ])
            ),
            (
                "get_current_temperature_v2",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "function") as (String, Any),
                    (
                        "function",
                        OrderedDictionary(uniqueKeysWithValues: [
                            ("name", "get_current_temperature") as (String, Any),
                            ("description", "Get the current temperature at a location.") as (String, Any),
                            (
                                "parameters",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "properties",
                                        OrderedDictionary(uniqueKeysWithValues: [
                                            (
                                                "location",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    (
                                                        "description",
                                                        "The location to get the temperature for, in the format \"City, Country\""
                                                    ) as (String, Any),
                                                ])
                                            ) as (String, Any),
                                            (
                                                "unit",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    ("enum", ["celsius", "fahrenheit"]) as (String, Any),
                                                    ("description", "The unit to return the temperature in.")
                                                        as (String, Any),
                                                ])
                                            ) as (String, Any),
                                        ])
                                    ) as (String, Any),
                                    ("required", ["location", "unit"]) as (String, Any),
                                ])
                            ) as (String, Any),
                            (
                                "return",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "number") as (String, Any),
                                    (
                                        "description",
                                        "The current temperature at the specified location in the specified units, as a float."
                                    ) as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                ])
            ),
            (
                "get_current_wind_speed",
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "function") as (String, Any),
                    (
                        "function",
                        OrderedDictionary(uniqueKeysWithValues: [
                            ("name", "get_current_wind_speed") as (String, Any),
                            ("description", "Get the current wind speed in km/h at a given location.") as (String, Any),
                            (
                                "parameters",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "properties",
                                        OrderedDictionary(uniqueKeysWithValues: [
                                            (
                                                "location",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    (
                                                        "description",
                                                        "The location to get the temperature for, in the format \"City, Country\""
                                                    ) as (String, Any),
                                                ])
                                            ) as (String, Any)
                                        ])
                                    ) as (String, Any),
                                    ("required", ["location"]) as (String, Any),
                                ])
                            ) as (String, Any),
                            (
                                "return",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "number") as (String, Any),
                                    ("description", "The current wind speed at the given location in km/h, as a float.")
                                        as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                ])
            ),
        ])

    lazy var exampleListOfTools: [OrderedDictionary<String, Any>] = [
        exampleToolJSONSchemas["get_current_temperature_v2"]!,
        exampleToolJSONSchemas["get_current_wind_speed"]!,
    ]

    func testMeetKaiFunctionaryMediumV2_2() throws {
        let chatTemplate = """
            {#v2.2#}\n{% for message in messages %}\n{% if message['role'] == 'user' or message['role'] == 'system' %}\n{{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}{% elif message['role'] == 'tool' %}\n{{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}{% else %}\n{% set contain_content='no'%}\n{% if message['content'] is not none %}\n{{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}{% set contain_content='yes'%}\n{% endif %}\n{% if 'tool_calls' in message and message['tool_calls'] is not none %}\n{% for tool_call in message['tool_calls'] %}\n{% set prompt='<|from|>assistant\n<|recipient|>' + tool_call['function']['name'] + '\n<|content|>' + tool_call['function']['arguments'] %}\n{% if loop.index == 1 and contain_content == "no" %}\n{{ prompt }}{% else %}\n{{ '\n' + prompt}}{% endif %}\n{% endfor %}\n{% endif %}\n{{ '<|stop|>\n' }}{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}{{ '<|from|>assistant\n<|recipient|>' }}{% endif %}
            """
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithFunctionCalling,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "add_generation_prompt": false,
        ])
        let target =
            """
            <|from|>assistant\n<|recipient|>get_current_weather\n<|content|>{\n  "location": "Hanoi"\n}<|stop|>\n<|from|>user\n<|recipient|>all\n<|content|>What's the weather like in Hanoi?\n
            """
        XCTAssertEqual(result, target)
    }

    func testFireworksAIFireFunctionV1() throws {
        let chatTemplate = """
                {%- set message_roles = ['SYSTEM', 'FUNCTIONS', 'USER', 'ASSISTANT', 'TOOL'] -%}\n{%- set ns = namespace(seen_non_system=false, messages=messages, content='', functions=[]) -%}\n{{ bos_token }}\n{#- Basic consistency checks -#}\n{%- if not ns.messages -%}\n  {{ raise_exception('No messages') }}\n{%- endif -%}\n{%- if ns.messages[0]['role'] | upper != 'SYSTEM' -%}\n  {%- set ns.messages = [{'role': 'SYSTEM', 'content': 'You are a helpful assistant with access to functions. Use them if required.'}] + ns.messages -%}\n{%- endif -%}\n{%- if ns.messages | length < 2 or ns.messages[0]['role'] | upper != 'SYSTEM' or ns.messages[1]['role'] | upper != 'FUNCTIONS' -%}\n  {{ raise_exception('Expected either "functions" or ["system", "functions"] as the first messages') }}\n{%- endif -%}\n{%- for message in ns.messages -%}\n  {%- set role = message['role'] | upper -%}\n  {#- Validation -#}\n  {%- if role not in message_roles -%}\n    {{ raise_exception('Invalid role ' + message['role'] + '. Only ' + message_roles + ' are supported.') }}\n  {%- endif -%}\n  {%- set ns.content = message['content'] if message.get('content') else '' -%}\n  {#- Move tool calls inside the content -#}\n  {%- if 'tool_calls' in message -%}\n    {%- for call in message['tool_calls'] -%}\n      {%- set ns.content = ns.content + '<functioncall>{"name": "' + call['function']['name'] + '", "arguments": ' + call['function']['arguments'] + '}' -%}\n    {%- endfor -%}\n  {%- endif -%}\n  {%- if role == 'ASSISTANT' and '<functioncall>' not in ns.content -%}\n    {%- set ns.content = '<plain>' + ns.content -%}\n  {%- endif -%}\n  {%- if role == 'ASSISTANT' -%}\n    {%- set ns.content = ns.content + eos_token -%}\n  {%- endif -%}\n  {{ role }}: {{ ns.content }}{{ '\\n\\n' }}\n{%- endfor -%}\nASSISTANT:{{ ' ' }}\n
            """
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithFunctionCallingAndSystemPrompt,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "add_generation_prompt": false,
        ])
        let target = """
            <s>SYSTEM: You are a helpful assistant with access to functions. Use them if required.

            FUNCTIONS: [{"name": "get_stock_price", "description": "Get the current stock price", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL, GOOG"}}, "required": ["symbol"]}}, {"name": "check_word_anagram", "description": "Check if two words are anagrams of each other", "parameters": {"type": "object", "properties": {"word1": {"type": "string", "description": "The first word"}, "word2": {"type": "string", "description": "The second word"}}, "required": ["word1", "word2"]}}]

            USER: Hi, can you tell me the current stock price of AAPL?

            ASSISTANT: 
            """
        XCTAssertEqual(result, target)
    }

    // Fails because tools are omitted in the output, and the result is indented.
    //        func testMistral7BInstructV0_3JSONSchema() throws {
    //            let chatTemplate =
    //                "{{- bos_token }}\n{%- set user_messages = messages | selectattr('role', 'equalto', 'user') | list %}\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {%- if tools and (message == user_messages[-1]) %}\n            {{- ' [AVAILABLE_TOOLS] [' }}\n            {%- for tool in tools %}\n\t\t{%- set tool = tool.function %}\n\t\t{{- '{\"type\": \"function\", \"function\": {' }}\n\t\t{%- for key, val in tool|items if key != \"return\" %}\n\t\t    {%- if val is string %}\n\t\t\t{{- '\"' + key + '\": \"' + val + '\"' }}\n\t\t    {%- else %}\n\t\t\t{{- '\"' + key + '\": ' + val|tojson }}\n\t\t    {%- endif %}\n\t\t    {%- if not loop.last %}\n\t\t\t{{- \", \" }}\n\t\t    {%- endif %}\n\t\t{%- endfor %}\n\t\t{{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- ' [/AVAILABLE_TOOLS]' }}\n            {%- endif %}\n        {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {%- if message.tool_calls is defined and message.tool_calls|length > 0 %}\n            {{- ' [TOOL_CALLS] [' }}\n            {%- for tool_call in message.tool_calls %}\n                {{- {\"name\": tool_call.function.name, \"arguments\": tool_call.function.arguments, \"id\": tool_call.id}|tojson }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- '] ' }}\n            {{- eos_token }}\n    \t{%- elif message.content is defined %}\n\t    {{- ' ' + message.content + ' ' + eos_token}}\n        {%- endif %}\n    {%- elif message['role'] == 'tool' %}\n        {{- ' [TOOL_RESULTS] ' }}\n        {{- '{\"call_id\": \"' + message.tool_call_id + '\", \"content\": ' + message.content|string + '}' }}\n        {{- ' [/TOOL_RESULTS] ' }}\n    {%- endif %}\n{%- endfor %}\n"
    //            let template = try Template(chatTemplate)
    //
    //            let result = try template.render([
    //                "messages": [
    //                    [
    //                        "role": "system",
    //                        "content":
    //                            "You are a bot that responds to weather queries. You should reply with the unit used in the queried location.",
    //                    ],
    //                    ["role": "user", "content": "Hey, what's the temperature in Paris right now?"],
    //                    [
    //                        "role": "assistant",
    //                        "tool_calls": [
    //                            [
    //                                "id": "abcdef123",
    //                                "type": "function",
    //                                "function": [
    //                                    "name": "get_current_temperature",
    //                                    "arguments": ["location": "Paris, France", "unit": "celsius"],
    //                                ],
    //                            ]
    //                        ],
    //                    ],
    //                    ["role": "tool", "tool_call_id": "abcdef123", "name": "get_current_temperature", "content": "22.0"],
    //                ],
    //                "tools": exampleListOfTools,
    //                // "tools_json": "", // TODO: Figure out how to convert the array of OrderedDictionaries to JSON
    //                "bos_token": "<s>",
    //                "eos_token": "</s>",
    //            ])
    //            let target = """
    //                <s> [AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_current_temperature", "description": "Get the current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, Country\\""}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in."}}, "required": ["location", "unit"]}}}, {"type": "function", "function": {"name": "get_current_wind_speed", "description": "Get the current wind speed in km/h at a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, Country\\""}}, "required": ["location"]}}}] [/AVAILABLE_TOOLS] [INST] Hey, what\'s the temperature in Paris right now? [/INST] [TOOL_CALLS] [{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}, "id": "abcdef123"}] </s> [TOOL_RESULTS] {"call_id": "abcdef123", "content": 22.0} [/TOOL_RESULTS]
    //                """
    //
    //            XCTAssertEqual(result, target)
    //        }

    // Previously failed because tools are omitted in the output, now fails because of error with `map`: runtime("map filter requires either an attribute name or a function")
    //    func testCISCaiMistral7BInstructV0_3SOTAGGUF() throws {
    //        let chatTemplate = """
    //              {{ bos_token }}{% set ns = namespace(lastuser=-1, system=false, functions=false) %}{% if tools %}{% for message in messages %}{% if message['role'] == 'user' %}{% set ns.lastuser = loop.index0 %}{% elif message['role'] == 'system' %}{% set ns.system = message['content'] %}{% endif %}{% endfor %}{% set ns.functions = tools|selectattr('type','eq','function')|map(attribute='function')|list|tojson %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.index0 == ns.lastuser and ns.functions %}{{ '[AVAILABLE_TOOLS] ' }}{{ ns.functions }}{{ '[/AVAILABLE_TOOLS]' }}{% endif %}{{ '[INST] ' }}{% if loop.index0 == ns.lastuser and ns.system %}{{ ns.system + ' ' }}{% endif %}{{ message['content'] }}{{ '[/INST]' }}{% elif message['role'] == 'tool' %}{{ '[TOOL_RESULTS] ' }}{{ dict(call_id=message['tool_call_id'], content=message['content'])|tojson }}{{ '[/TOOL_RESULTS]' }}{% elif message['role'] == 'assistant' %}{% if message['tool_calls'] %}{{ '[TOOL_CALLS] [' }}{% for call in message['tool_calls'] %}{% if call['type'] == 'function' %}{{ dict(id=call['id'], name=call['function']['name'], arguments=call['function']['arguments'])|tojson }}{% endif %}{% if not loop.last %}{{ ', ' }}{% endif %}{% endfor %}{{ ']' }}{% else %}{{ message['content'] }}{% endif %}{{ eos_token }}{% endif %}{% endfor %}
    //            """
    //        let template = try Template(chatTemplate)
    //
    //        let result = try template.render([
    //            "messages": [
    //                [
    //                    "role": "user",
    //                    "content": "What's the weather like in Oslo and Stockholm?",
    //                ]
    //            ],
    //            "tools": [exampleToolJSONSchemas["get_current_temperature_v2"]!],
    //            "bos_token": "<s>",
    //            "eos_token": "</s>",
    //        ])
    //        let target =
    //            """
    //            <s>[AVAILABLE_TOOLS] [{"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}][/AVAILABLE_TOOLS][INST] What's the weather like in Oslo and Stockholm?[/INST]
    //            """
    //
    //        XCTAssertEqual(result, target)
    //    }

    func testNousResearchHermes2ProLlama38BJSONSchema() throws {
        let chatTemplate = """
            {%- macro json_to_python_type(json_spec) %}\n{%- set basic_type_map = {\n    "string": "str",\n    "number": "float",\n    "integer": "int",\n    "boolean": "bool"\n} %}\n\n{%- if basic_type_map[json_spec.type] is defined %}\n    {{- basic_type_map[json_spec.type] }}\n{%- elif json_spec.type == "array" %}\n    {{- "list[" +  json_to_python_type(json_spec|items) + "]"}}\n{%- elif json_spec.type == "object" %}\n    {%- if json_spec.additionalProperties is defined %}\n        {{- "dict[str, " + json_to_python_type(json_spec.additionalProperties) + ']'}}\n    {%- else %}\n        {{- "dict" }}\n    {%- endif %}\n{%- elif json_spec.type is iterable %}\n    {{- "Union[" }}\n    {%- for t in json_spec.type %}\n      {{- json_to_python_type({"type": t}) }}\n      {%- if not loop.last %}\n        {{- "," }} \n    {%- endif %}\n    {%- endfor %}\n    {{- "]" }}\n{%- else %}\n    {{- "Any" }}\n{%- endif %}\n{%- endmacro %}\n\n\n{{- bos_token }}\n{{- "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> " }}\n{%- for tool in tools %}\n    {%- if tool.function is defined %}\n        {%- set tool = tool.function %}\n    {%- endif %}\n    {{- '{"type": "function", "function": ' }}\n    {{- '{"name": ' + tool.name + '", ' }}\n    {{- '"description": "' + tool.name + '(' }}\n    {%- for param_name, param_fields in tool.parameters.properties|items %}\n        {{- param_name + ": " + json_to_python_type(param_fields) }}\n        {%- if not loop.last %}\n            {{- ", " }}\n        {%- endif %}\n    {%- endfor %}\n    {{- ")" }}\n    {%- if tool.return is defined %}\n        {{- " -> " + json_to_python_type(tool.return) }}\n    {%- endif %}\n    {{- " - " + tool.description + "\\n\\n" }}\n    {%- for param_name, param_fields in tool.parameters.properties|items %}\n        {%- if loop.first %}\n            {{- "    Args:\\n" }}\n        {%- endif %}\n        {{- "        " + param_name + "(" + json_to_python_type(param_fields) + "): " + param_fields.description|trim }}\n    {%- endfor %}\n    {%- if tool.return is defined and tool.return.description is defined %}\n        {{- "\\n    Returns:\\n        " + tool.return.description }}\n    {%- endif %}\n    {{- '"' }}\n    {{- ', "parameters": ' }}\n    {%- if tool.parameters.properties | length == 0 %}\n        {{- "{}" }}\n    {%- else %}\n        {{- tool.parameters | tojson}}\n    {%- endif %}\n    {{- "}" }}\n    {%- if not loop.last %}\n        {{- "\\n" }}\n    {%- endif %}\n{%- endfor %}\n{{- " </tools>" }}\n{{- 'Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}\n' }}\n{{- "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n" }}\n{{- "<tool_call>\n" }}\n{{- '{"arguments": <args-dict>, "name": <function-name>}\n' }}\n{{- '</tool_call><|im_end|>' }}\n{%- for message in messages %}\n    {%- if message.role == "user" or message.role == "system" or (message.role == "assistant" and message.tool_calls is not defined) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == "assistant" %}\n        {{- '<|im_start|>' + message.role + '\\n<tool_call>\\n' }}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '{ ' }}\n            {%- if tool_call.arguments is defined %}\n                {{- '"arguments": ' }}\n                {{- tool_call.arguments|tojson }}\n                {{- ', '}}\n            {%- endif %}\n            {{- '"name": "' }}\n            {{- tool_call.name }}\n            {{- '"}' }}\n            {{- '\\n</tool_call> ' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == "tool" %}\n        {%- if not message.name is defined %}\n            {{- raise_exception("Tool response dicts require a 'name' key indicating the name of the called function!") }}\n        {%- endif %}\n        {{- '<|im_start|>' + message.role + '\\n<tool_response>\\n' }}\n        {{- '{"name": "' }}\n        {{- message.name }}\n        {{- '", "content": ' }}\n        {{- message.content|tojson + '}' }}\n        {{- '\\n</tool_response> <|im_end|>\\n' }} \n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
            """
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": [
                OrderedDictionary(uniqueKeysWithValues: [
                    ("role", "user") as (String, Any),
                    ("content", "Fetch the stock fundamentals data for Tesla (TSLA)") as (String, Any),
                ])
            ],
            "tools": [
                OrderedDictionary(uniqueKeysWithValues: [
                    ("type", "function") as (String, Any),
                    (
                        "function",
                        OrderedDictionary(uniqueKeysWithValues: [
                            ("name", "get_stock_fundamentals") as (String, Any),
                            ("description", "Get fundamental data for a given stock symbol using yfinance API.")
                                as (String, Any),
                            (
                                "parameters",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "properties",
                                        OrderedDictionary(uniqueKeysWithValues: [
                                            (
                                                "symbol",
                                                OrderedDictionary(uniqueKeysWithValues: [
                                                    ("type", "string") as (String, Any),
                                                    ("description", "The stock symbol.") as (String, Any),
                                                ])
                                            ) as (String, Any)
                                        ])
                                    ) as (String, Any),
                                    ("required", ["symbol"]) as (String, Any),
                                ])
                            ) as (String, Any),
                            (
                                "return",
                                OrderedDictionary(uniqueKeysWithValues: [
                                    ("type", "object") as (String, Any),
                                    (
                                        "description",
                                        """
                                        A dictionary containing fundamental data.

                                        Keys:
                                            - 'symbol': The stock symbol.
                                            - 'company_name': The long name of the company.
                                            - 'sector': The sector to which the company belongs.
                                            - 'industry': The industry to which the company belongs.
                                            - 'market_cap': The market capitalization of the company.
                                            - 'pe_ratio': The forward price-to-earnings ratio.
                                            - 'pb_ratio': The price-to-book ratio.
                                            - 'dividend_yield': The dividend yield.
                                            - 'eps': The trailing earnings per share.
                                            - 'beta': The beta value of the stock.
                                            - '52_week_high': The 52-week high price of the stock.
                                            - '52_week_low': The 52-week low price of the stock.
                                        """
                                    ) as (String, Any),
                                ])
                            ) as (String, Any),
                        ])
                    ) as (String, Any),
                ])
            ],
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|im_end|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|>You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\n\n    Args:\n        symbol(str): The stock symbol.\n    Returns:\n        A dictionary containing fundamental data.\n\nKeys:\n    - 'symbol': The stock symbol.\n    - 'company_name': The long name of the company.\n    - 'sector': The sector to which the company belongs.\n    - 'industry': The industry to which the company belongs.\n    - 'market_cap': The market capitalization of the company.\n    - 'pe_ratio': The forward price-to-earnings ratio.\n    - 'pb_ratio': The price-to-book ratio.\n    - 'dividend_yield': The dividend yield.\n    - 'eps': The trailing earnings per share.\n    - 'beta': The beta value of the stock.\n    - '52_week_high': The 52-week high price of the stock.\n    - '52_week_low': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "The stock symbol."}}, "required": ["symbol"]}} </tools>Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}\nFor each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|><|im_start|>user\nFetch the stock fundamentals data for Tesla (TSLA)<|im_end|>\n<|im_start|>assistant\n
            """
        XCTAssertEqual(result, target)
    }

    //    func testMetaLlamaLlama3_18BInstruct() throws {
    //        let chatTemplate = """
    //            {{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '="' + arg_val + '"' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{"name": "' + tool_call.name + '", ' }}\n            {{- '"parameters": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
    //            """
    //        let template = try Template(chatTemplate)
    //        let result = try template.render([
    //            "messages": [
    //                ["role": "system", "content": "You are a bot that responds to weather queries."],
    //                ["role": "user", "content": "Hey, what's the temperature in Paris right now?"],
    //            ],
    //            "tools": [exampleToolJSONSchemas["get_current_temperature_v1"]!],
    //            "bos_token": "<|begin_of_text|>",
    //            "eos_token": "<|im_end|>",
    //            "add_generation_prompt": true,
    //        ])
    //        let target = """
    //            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a bot that responds to weather queries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.\n\n{\n    "type": "function",\n    "function": {\n        "name": "get_current_temperature",\n        "description": "Get the current temperature at a location.",\n        "parameters": {\n            "type": "object",\n            "properties": {\n                "location": {\n                    "type": "string",\n                    "description": "The location to get the temperature for, in the format \\"City, Country\\""\n                }\n            },\n            "required": [\n                "location"\n            ]\n        },\n        "return": {\n            "type": "number",\n            "description": "The current temperature at the specified location in the specified units, as a float."\n        }\n    }\n}\n\nHey, what's the temperature in Paris right now?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    //            """
    //        XCTAssertEqual(result, target)
    //    }

    //

    func testLlama3_1() throws {
        let chatTemplate = ChatTemplate.llama3_1
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": Messages.weatherQuery,
            "tools": [ToolSpec.getCurrentWeather],
            "bos_token": "<|begin_of_text|>",
            //      "eos_token": "<|im_end|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            Environment: ipython
            Cutting Knowledge Date: December 2023
            Today Date: 26 Jul 2024

            <|eot_id|><|start_header_id|>user<|end_header_id|>

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

            What is the weather in Paris today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


            """
        XCTAssertEqual(result, target)
    }

    func testLlama3_2() throws {
        let chatTemplate = ChatTemplate.llama3_2
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": Messages.weatherQuery,
            "tools": [ToolSpec.getCurrentWeather],
            "bos_token": "<|begin_of_text|>",
            //      "eos_token": "<|im_end|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            Environment: ipython
            Cutting Knowledge Date: December 2023
            Today Date: \(Environment.formatDate(Date(), withFormat: "%d %b %Y"))

            <|eot_id|><|start_header_id|>user<|end_header_id|>

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

            What is the weather in Paris today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


            """
        XCTAssertEqual(result, target)
    }

    func testQwen2_5() throws {
        let chatTemplate = ChatTemplate.qwen2_5
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": Messages.weatherQuery,
            "tools": [ToolSpec.getCurrentWeather],
            "bos_token": "<|begin_of_text|>",
            //      "eos_token": "<|im_end|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>system
            You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            {"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call><|im_end|>
            <|im_start|>user
            What is the weather in Paris today?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testMistral7b() throws {
        let chatTemplate = ChatTemplate.mistral7b
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": Messages.weatherQuery,
            "tools": [ToolSpec.getCurrentWeather],
            "bos_token": "<|begin_of_text|>",
            //      "eos_token": "<|im_end|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}][/AVAILABLE_TOOLS][INST]What is the weather in Paris today?[/INST]
            """
        XCTAssertEqual(result, target)
    }
}

extension Data {
    var string: String? {
        return String(data: self, encoding: .utf8)
    }
}
