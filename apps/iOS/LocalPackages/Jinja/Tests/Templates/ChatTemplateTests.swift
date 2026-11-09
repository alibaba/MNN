//
//  ChatTemplateTests.swift
//
//
//  Created by John Mai on 2024/3/24.
//

import XCTest
import OrderedCollections

@testable import Jinja

final class ChatTemplateTests: XCTestCase {
    let messages: [[String: String]] = [
        [
            "role": "user",
            "content": "Hello, how are you?",
        ],
        [
            "role": "assistant",
            "content": "I'm doing great. How can I help you today?",
        ],
        [
            "role": "user",
            "content": "I'd like to show off how chat templating works!",
        ],
    ]

    let systemPromptMessage: [String: String] = [
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    ]

    lazy var messagesWithSystemPrompt: [[String: String]] = [systemPromptMessage] + messages

    func testGenericChatTemplate() throws {
        let chatTemplate =
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "add_generation_prompt": false,
        ])
        let target =
            "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testFacebookBlenderbot400MDistill() throws {
        let chatTemplate =
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "</s>",
        ])
        let target =
            " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
        XCTAssertEqual(result, target)
    }

    func testFacebookBlenderbotSmall90M() throws {
        let chatTemplate =
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "</s>",
        ])
        let target =
            " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
        XCTAssertEqual(result, target)
    }

    func testBigscienceBloom() throws {
        let chatTemplate = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "</s>",
        ])
        let target =
            "Hello, how are you?</s>I'm doing great. How can I help you today?</s>I'd like to show off how chat templating works!</s>"
        XCTAssertEqual(result, target)
    }

    func testEleutherAIGptNeox20b() throws {
        let chatTemplate = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "<|endoftext|>",
        ])
        let target =
            "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>"
        XCTAssertEqual(result, target)
    }

    func testGPT2() throws {
        let chatTemplate = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "<|endoftext|>",
        ])
        let target =
            "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>"
        XCTAssertEqual(result, target)
    }

    func testHfInternalTestingLlamaTokenizer1() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithSystemPrompt,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "USE_DEFAULT_PROMPT": true,
        ])
        let target =
            "<s>[INST] <<SYS>>\nYou are a friendly chatbot who always responds in the style of a pirate\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testHfInternalTestingLlamaTokenizer2() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "USE_DEFAULT_PROMPT": true,
        ])
        let target =
            "<s>[INST] <<SYS>>\nDEFAULT_SYSTEM_MESSAGE\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testHfInternalTestingLlamaTokenizer3() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": [
                [
                    "role": "user",
                    "content": "<<SYS>>\nYou are a helpful assistant\n<</SYS>> Hello, how are you?",
                ],
                [
                    "role": "assistant",
                    "content": "I'm doing great. How can I help you today?",
                ],
                [
                    "role": "user",
                    "content": "I'd like to show off how chat templating works!",
                ],
            ],
            "bos_token": "<s>",
            "eos_token": "</s>",
            "USE_DEFAULT_PROMPT": true,
        ])
        let target =
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>> Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testOpenaiWhisperLargeV3() throws {
        let chatTemplate = "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "eos_token": "<|endoftext|>",
        ])
        let target =
            "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>"
        XCTAssertEqual(result, target)
    }

    func testQwenQwen1_5_1_8BChat1() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "add_generation_prompt": true,
        ])
        let target =
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI\'m doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI\'d like to show off how chat templating works!<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(result, target)
    }

    func testQwenQwen1_5_1_8BChat2() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithSystemPrompt,
            "add_generation_prompt": true,
        ])
        let target =
            "<|im_start|>system\nYou are a friendly chatbot who always responds in the style of a pirate<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI\'m doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI\'d like to show off how chat templating works!<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(result, target)
    }

    func testQwenQwen1_5_1_8BChat3() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithSystemPrompt
        ])
        let target =
            "<|im_start|>system\nYou are a friendly chatbot who always responds in the style of a pirate<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI\'m doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI\'d like to show off how chat templating works!"
        XCTAssertEqual(result, target)
    }

    func testTHUDMChatglm36b() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.first %}[gMASK]sop<|{{ message['role'] }}|>\n {{ message['content'] }}{% else %}<|{{ message['role'] }}|>\n {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messagesWithSystemPrompt
        ])
        let target =
            "[gMASK]sop<|system|>\n You are a friendly chatbot who always responds in the style of a pirate<|user|>\n Hello, how are you?<|assistant|>\n I\'m doing great. How can I help you today?<|user|>\n I\'d like to show off how chat templating works!"
        XCTAssertEqual(result, target)
    }

    func testGoogleGemma2bIt() throws {
        let chatTemplate =
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages
        ])
        let target =
            "<start_of_turn>user\nHello, how are you?<end_of_turn>\n<start_of_turn>model\nI\'m doing great. How can I help you today?<end_of_turn>\n<start_of_turn>user\nI\'d like to show off how chat templating works!<end_of_turn>\n"
        XCTAssertEqual(result, target)
    }

    func testQwenQwen2_5_0_5BInstruct() throws {
        let chatTemplate =
            "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages
        ])
        let target =
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI\'m doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI\'d like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testHuggingFaceH4Zephyr7bBetaAddGenerationPromptFalse() throws {
        let chatTemplate =
            "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messagesWithSystemPrompt, "eos_token": "</s>",
                "add_generation_prompt": false,
            ] as [String: Any]
        )
        let target =
            "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\nHello, how are you?</s>\n<|assistant|>\nI'm doing great. How can I help you today?</s>\n<|user|>\nI'd like to show off how chat templating works!</s>\n"
        XCTAssertEqual(result, target)
    }

    func testHuggingFaceH4Zephyr7bBetaAddGenerationPromptTrue() throws {
        let chatTemplate =
            "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": [
                    [
                        "role": "system",
                        "content": "You are a friendly chatbot who always responds in the style of a pirate",
                    ],
                    ["role": "user", "content": "How many helicopters can a human eat in one sitting?"],
                ], "eos_token": "</s>", "add_generation_prompt": true,
            ] as [String: Any]
        )
        let target =
            "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\nHow many helicopters can a human eat in one sitting?</s>\n<|assistant|>\n"
        XCTAssertEqual(result, target)
    }

    func testHuggingFaceH4Zephyr7bGemmaV0_1() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<bos>", "eos_token": "<eos>",
                "add_generation_prompt": false,
            ] as [String: Any]
        )
        let target =
            "<bos><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testTheBlokeMistral7BInstructV0_1GPTQ() throws {
        let chatTemplate =
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<s>", "eos_token": "</s>",
            ] as [String: Any]
        )
        let target =
            "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testMistralaiMixtral8x7BInstructV0_1() throws {
        let chatTemplate =
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<s>", "eos_token": "</s>",
            ] as [String: Any]
        )
        let target =
            "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s>[INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testCognitivecomputationsDolphin2_5Mixtral8x7b() throws {
        let chatTemplate =
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testOpenchatOpenchat3_5_0106() throws {
        let chatTemplate =
            "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<s>", "eos_token": "</s>",
                "add_generation_prompt": false,
            ] as [String: Any]
        )
        let target =
            "<s>GPT4 Correct User: Hello, how are you?<|end_of_turn|>GPT4 Correct Assistant: I'm doing great. How can I help you today?<|end_of_turn|>GPT4 Correct User: I'd like to show off how chat templating works!<|end_of_turn|>"
        XCTAssertEqual(result, target)
    }

    func testUpstageSOLAR10_7BInstructV1_0() throws {
        let chatTemplate =
            "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'### System:\n' + message['content']+'\n\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'### User:\n' + message['content']+'\n\n'}}{% elif message['role'] == 'assistant' %}{{'### Assistant:\n'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "### User:\nHello, how are you?\n\n### Assistant:\nI'm doing great. How can I help you today?### User:\nI'd like to show off how chat templating works!\n\n"
        XCTAssertEqual(result, target)
    }

    func testCodellamaCodeLlama70bInstructHf() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'system' %}{% set user_index = 1 %}{% else %}{% set user_index = 0 %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ '<s>' }}{% endif %}{% set content = 'Source: ' + message['role'] + '\n\n ' + message['content'] | trim %}{{ content + ' <step> ' }}{% endfor %}{{'Source: assistant\nDestination: user\n\n '}}";
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "<s>Source: user\n\n Hello, how are you? <step> Source: assistant\n\n I'm doing great. How can I help you today? <step> Source: user\n\n I'd like to show off how chat templating works! <step> Source: assistant\nDestination: user\n\n "
        XCTAssertEqual(result, target)
    }

    func testDeciDeciLM7BInstruct() throws {
        let chatTemplate =
            "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### User:\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '### System:\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '### Assistant:\n'  + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### Assistant:' }}\n{% endif %}\n{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "### User:\nHello, how are you?\n### Assistant:\nI'm doing great. How can I help you today?\n### User:\nI'd like to show off how chat templating works!\n"
        XCTAssertEqual(result, target)
    }

    func testQwenQwen1_5_72BChat() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testDeepseekAiDeepseekLlm7bChat() throws {
        let chatTemplate =
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<｜begin of sentence｜>",
                "eos_token": "<｜end of sentence｜>",
            ] as [String: Any]
        )
        let target =
            "<｜begin of sentence｜>User: Hello, how are you?\n\nAssistant: I'm doing great. How can I help you today?<｜end of sentence｜>User: I'd like to show off how chat templating works!\n\n"
        XCTAssertEqual(result, target)
    }

    func testH2oaiH2oDanube1_8bChat() throws {
        let chatTemplate =
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompt|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|answer|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|answer|>' }}{% endif %}{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "eos_token": "</s>",
            ] as [String: Any]
        )
        let target =
            "<|prompt|>Hello, how are you?</s><|answer|>I'm doing great. How can I help you today?</s><|prompt|>I'd like to show off how chat templating works!</s>"
        XCTAssertEqual(result, target)
    }

    func testInternlmInternlm2Chat7b() throws {
        let chatTemplate =
            "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<s>", "eos_token": "</s>",
            ] as [String: Any]
        )
        let target =
            "<s><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n"
        XCTAssertEqual(result, target)
    }

    func testTheBlokedeepseekCoder33BInstructAWQ() throws {
        let chatTemplate =
            "{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response:\\n'}}\n"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n### Response:\n"
        XCTAssertEqual(result, target)
    }

    func testEriczzzFalconRw1bChat() throws {
        let chatTemplate =
            "{% for message in messages %}{% if loop.index > 1 and loop.previtem['role'] != 'assistant' %}{{ ' ' }}{% endif %}{% if message['role'] == 'system' %}{{ '[SYS] ' + message['content'].strip() }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '[RESP] '  + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' [RESP] ' }}{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "eos_token": "<|endoftext|>",
            ] as [String: Any]
        )
        let target =
            "[INST] Hello, how are you? [RESP] I'm doing great. How can I help you today?<|endoftext|>[INST] I'd like to show off how chat templating works!"
        XCTAssertEqual(result, target)
    }

    func testAbacusaiSmaug34BV0_1() throws {
        let chatTemplate =
            "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<s>", "eos_token": "</s>",
            ] as [String: Any]
        )
        let target =
            "Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
        XCTAssertEqual(result, target)
    }

    func testMaywellSynatraMixtral8x7B() throws {
        let chatTemplate =
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'system' %}{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n### Response:\n{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages
            ] as [String: Any]
        )
        let target =
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nHello, how are you?### Response:\nI'm doing great. How can I help you today?### Instruction:\nI'd like to show off how chat templating works!"
        XCTAssertEqual(result, target)
    }

    func testDeepseekAiDeepseekCoder33bInstruct() throws {
        let chatTemplate =
            "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messages, "bos_token": "<｜begin of sentence｜>", "eos_token": "<|EOT|>",
            ] as [String: Any]
        )
        let target =
            "<｜begin of sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n"
        XCTAssertEqual(result, target)
    }

    func testMaywellSynatraMixtral8x7B_2() throws {
        let chatTemplate =
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'system' %}{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n### Response:\n{% endif %}"
        let template = try Template(chatTemplate)
        let result = try template.render(
            [
                "messages": messagesWithSystemPrompt
            ] as [String: Any]
        )
        let target =
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nYou are a friendly chatbot who always responds in the style of a pirate### Instruction:\nHello, how are you?### Response:\nI'm doing great. How can I help you today?### Instruction:\nI'd like to show off how chat templating works!"
        XCTAssertEqual(result, target)
    }

    func testMistralNemoInstruct2407() throws {
        let chatTemplate =
            "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{%- for message in loop_messages | rejectattr(\"role\", \"equalto\", \"tool\") | rejectattr(\"role\", \"equalto\", \"tool_results\") | selectattr(\"tool_calls\", \"undefined\") %}\n    {%- if (message[\"role\"] == \"user\") != (loop.index0 % 2 == 0) %}\n        {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS][\" }}\n            {%- for tool in tools %}\n        {%- set tool = tool.function %}\n        {{- '{\"type\": \"function\", \"function\": {' }}\n        {%- for key, val in tool.items() if key != \"return\" %}\n            {%- if val is string %}\n            {{- '\"' + key + '\": \"' + val + '\"' }}\n            {%- else %}\n            {{- '\"' + key + '\": ' + val|tojson }}\n            {%- endif %}\n            {%- if not loop.last %}\n            {{- \", \" }}\n            {%- endif %}\n        {%- endfor %}\n        {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif message[\"role\"] == \"tool_calls\" or message.tool_calls is defined %}\n        {%- if message.tool_calls is defined %}\n            {%- set tool_calls = message.tool_calls %}\n        {%- else %}\n            {%- set tool_calls = message.content %}\n        {%- endif %}\n        {{- \"[TOOL_CALLS][\" }}\n        {%- for tool_call in tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS]{\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n"
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": messages,
            "bos_token": "<s>",
            "eos_token": "</s>",
        ])
        let target =
            "<s>[INST]Hello, how are you?[/INST]I'm doing great. How can I help you today?</s>[INST]I'd like to show off how chat templating works![/INST]"

        XCTAssertEqual(result, target)
    }

    func testQwen2VLTextOnly() throws {
        let qwen2VLChatTemplate =
            "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        let template = try Template(qwen2VLChatTemplate)
        let result = try template.render([
            "messages": messages,
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            Hello, how are you?<|im_end|>
            <|im_start|>assistant
            I'm doing great. How can I help you today?<|im_end|>
            <|im_start|>user
            I'd like to show off how chat templating works!<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testPhi4() throws {
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let chatTemplate = """
            {% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}
            """
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>user<|im_sep|>What is the weather in Paris today?<|im_end|><|im_start|>assistant<|im_sep|>
            """
        XCTAssertEqual(result, target)
    }

    let deepSeekR1chatTemplate = """
        {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}
        """

    func testDeepSeekR1() throws {
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let template = try Template(deepSeekR1chatTemplate)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|><｜User｜>What is the weather in Paris today?<｜Assistant｜>
            """
        XCTAssertEqual(result, target)
    }

    func testDeepSeekR1WithSystemPrompt() throws {
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let template = try Template(deepSeekR1chatTemplate)
        let result = try template.render([
            "messages": [systemPromptMessage, userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|begin_of_text|>You are a friendly chatbot who always responds in the style of a pirate<｜User｜>What is the weather in Paris today?<｜Assistant｜>
            """
        XCTAssertEqual(result, target)
    }

    func testQwen3() throws {
        let chatTemplate = """
            {%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}
            """
        let userMessage = [
            "role": "user",
            "content": "Why is the sky blue?",
        ]
        let template = try Template(chatTemplate)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>user
            Why is the sky blue?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testGraniteWithoutThinking() throws {
        let userMessage = [
            "role": "user",
            "content": "What is 1+1?",
        ]
        let template = try Template(ChatTemplate.granite3_3)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
            Today's Date: \(Environment.formatDate(Date(), withFormat: "%B %d, %Y")).
            You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
            <|start_of_role|>user<|end_of_role|>What is 1+1?<|end_of_text|>
            <|start_of_role|>assistant<|end_of_role|>
            """
        XCTAssertEqual(result, target)
    }

    func testGraniteWithThinking() throws {
        let userMessage = [
            "role": "user",
            "content": "What is 1+1?",
        ]
        let template = try Template(ChatTemplate.granite3_3)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
            "thinking": true,
        ])
        let target = """
            <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
            Today's Date: \(Environment.formatDate(Date(), withFormat: "%B %d, %Y")).
            You are Granite, developed by IBM. You are a helpful AI assistant.
            Respond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query.<|end_of_text|>
            <|start_of_role|>user<|end_of_role|>What is 1+1?<|end_of_text|>
            <|start_of_role|>assistant<|end_of_role|>
            """
        XCTAssertEqual(result, target)
    }

    func testSmolLM3() throws {
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let template = try Template(ChatTemplate.smollm3)
        let result = try template.render([
            "messages": [userMessage],
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>system
            ## Metadata

            Knowledge Cutoff Date: June 2025
            Today Date: \(Environment.formatDate(Date(), withFormat: "%d %B %Y"))
            Reasoning Mode: /think

            ## Custom Instructions

            You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

            <|im_start|>user
            What is the weather in Paris today?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testSmolLM3FromHF() throws {
        let data = try? Data(
            contentsOf: URL(string: "https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/chat_template.jinja")!
        )
        let chatTemplate = String(data: data!, encoding: .utf8)
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let template = try Template(chatTemplate!)
        let result = try template.render([
            "messages": [userMessage],
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>system
            ## Metadata

            Knowledge Cutoff Date: June 2025
            Today Date: \(Environment.formatDate(Date(), withFormat: "%d %B %Y"))
            Reasoning Mode: /think

            ## Custom Instructions

            You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

            <|im_start|>user
            What is the weather in Paris today?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testSmolLM3WithSystemPrompt() throws {
        let data = try? Data(
            contentsOf: URL(string: "https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/chat_template.jinja")!
        )
        let chatTemplate = String(data: data!, encoding: .utf8)
        let systemMessage = [
            "role": "system",
            "content": "You are a assistant.",
        ]
        let userMessage = [
            "role": "user",
            "content": "What is the weather in Paris today?",
        ]
        let template = try Template(chatTemplate!)
        let result = try template.render([
            "messages": [
                systemMessage,
                userMessage,
            ],
            "add_generation_prompt": true,
            "eos_token": "<|im_end|>",
            "pad_token": "<|im_end|>",
        ])

        let target = """
            <|im_start|>system
            ## Metadata

            Knowledge Cutoff Date: June 2025
            Today Date: \(Environment.formatDate(Date(), withFormat: "%d %B %Y"))
            Reasoning Mode: /think

            ## Custom Instructions

            You are a assistant.

            <|im_start|>user
            What is the weather in Paris today?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testQwen3Coder() throws {
        let userMessage = [
            "role": "user",
            "content": "Why is the sky blue?",
        ]
        let template = try Template(ChatTemplate.qwen3_coder)
        let result = try template.render([
            "messages": [userMessage],
            "bos_token": "<|begin_of_text|>",
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>user
            Why is the sky blue?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testQwen3CoderToolCallsAndResponse() throws {
        let toolCallMessages = [
            OrderedDictionary<String, Any>(
                dictionaryLiteral: ("role", "user"),
                ("content", "What's the weather in Shenzhen?")
            ),
            OrderedDictionary<String, Any>(
                dictionaryLiteral: ("role", "assistant"),
                ("content", "I'll check the weather in Shenzhen for you."),
                (
                    "tool_calls",
                    [
                        OrderedDictionary<String, Any>(
                            dictionaryLiteral: (
                                "function",
                                OrderedDictionary<String, Any>(
                                    dictionaryLiteral: ("name", "get_weather"),
                                    (
                                        "arguments",
                                        OrderedDictionary<String, Any>(
                                            dictionaryLiteral: ("location", "Shenzhen, China"),
                                            ("unit", "celsius")
                                        )
                                    )
                                )
                            )
                        )
                    ]
                )
            ),
            OrderedDictionary<String, Any>(
                dictionaryLiteral: ("role", "tool"),
                ("content", "The current weather in Shenzhen, China is 22°C with clear skies.")
            ),
        ]
        let template = try Template(ChatTemplate.qwen3_coder)
        let result = try template.render([
            "messages": toolCallMessages,
            "add_generation_prompt": true,
        ])
        let target = """
            <|im_start|>user
            What's the weather in Shenzhen?<|im_end|>
            <|im_start|>assistant
            I'll check the weather in Shenzhen for you.

            <tool_call>
            <function=get_weather>
            <parameter=location>
            Shenzhen, China
            </parameter>
            <parameter=unit>
            celsius
            </parameter>
            </function>
            </tool_call><|im_end|>
            <|im_start|>user
            <tool_response>
            The current weather in Shenzhen, China is 22°C with clear skies.
            </tool_response>
            <|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testQwen3CoderWithEnum() throws {
        let toolWithEnum: OrderedDictionary<String, Any> = [
            "type": "function",
            "function": OrderedDictionary<String, Any>(
                dictionaryLiteral: ("name", "get_weather"),
                ("description", "Get weather information for a specified location"),
                (
                    "parameters",
                    OrderedDictionary<String, Any>(
                        dictionaryLiteral: ("type", "object"),
                        (
                            "properties",
                            OrderedDictionary<String, Any>(
                                dictionaryLiteral: (
                                    "location",
                                    OrderedDictionary<String, Any>(
                                        dictionaryLiteral: ("type", "string"),
                                        ("description", "City and state, e.g. San Francisco, CA")
                                    )
                                ),
                                (
                                    "unit",
                                    OrderedDictionary<String, Any>(
                                        dictionaryLiteral: ("type", "string"),
                                        ("description", "Temperature unit"),
                                        ("enum", ["celsius", "fahrenheit"])
                                    )
                                ),
                                (
                                    "format",
                                    OrderedDictionary<String, Any>(
                                        dictionaryLiteral: ("type", "string"),
                                        ("description", "Return format"),
                                        ("enum", ["json", "text", "xml"])
                                    )
                                )
                            )
                        ),
                        ("required", ["location"])
                    )
                )
            ),
        ]

        let userMessage = [
            "role": "user",
            "content": "What's the weather like in Beijing today?",
        ]

        let template = try Template(ChatTemplate.qwen3_coder)
        let result = try template.render([
            "messages": [userMessage],
            "tools": [toolWithEnum],
            "add_generation_prompt": true,
        ])

        let target = """
            <|im_start|>system
            You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.

            You have access to the following functions:

            <tools>
            <function>
            <name>get_weather</name>
            <description>Get weather information for a specified location</description>
            <parameters>
            <parameter>
            <name>location</name>
            <type>"string"</type>
            <description>City and state, e.g. San Francisco, CA</description>
            </parameter>
            <parameter>
            <name>unit</name>
            <type>"string"</type>
            <description>Temperature unit</description>
            <enum>[`celsius`, `fahrenheit`]</enum>
            </parameter>
            <parameter>
            <name>format</name>
            <type>"string"</type>
            <description>Return format</description>
            <enum>[`json`, `text`, `xml`]</enum>
            </parameter>
            <required>[`location`]</required>
            </parameters>
            </function>
            </tools>

            If you choose to call a function ONLY reply in the following format with NO suffix:

            <tool_call>
            <function=example_function_name>
            <parameter=example_parameter_1>
            value_1
            </parameter>
            <parameter=example_parameter_2>
            This is the value for the second parameter
            that can span
            multiple lines
            </parameter>
            </function>
            </tool_call>

            <IMPORTANT>
            Reminder:
            - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
            - Required parameters MUST be specified
            - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
            - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
            </IMPORTANT><|im_end|>
            <|im_start|>user
            What's the weather like in Beijing today?<|im_end|>
            <|im_start|>assistant

            """
        XCTAssertEqual(result, target)
    }

    func testReject() throws {
        let template = try Template(
            #"""
            {%- set handled_keys = ['type', 'description', 'enum', 'required'] %}
            {%- set all_keys = ['type', 'name', 'enum', 'value', 'description', 'age', 'required', 'status'] %}

            Filtered keys: {{ all_keys | reject("in", handled_keys) | list }}
            """#
        )
        let result = try template.render([:])
        print(result)
    }
}
