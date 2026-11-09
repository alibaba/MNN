//
//  ChatTemplateTests.swift
//  swift-transformers
//
//  Created by Anthony DePasquale on 2/10/24.
//

import Tokenizers
import XCTest

class ChatTemplateTests: XCTestCase {
    let messages = [[
        "role": "user",
        "content": "Describe the Swift programming language.",
    ]]

    func testTemplateFromConfig() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [32010, 4002, 29581, 278, 14156, 8720, 4086, 29889, 32007, 32001]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<|user|>Describe the Swift programming language.<|end|><|assistant|>"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testDeepSeekQwenChatTemplate() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [151646, 151644, 74785, 279, 23670, 15473, 4128, 13, 151645, 151648, 198]
        XCTAssertEqual(encoded, encodedTarget)

        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<｜begin▁of▁sentence｜><｜User｜>Describe the Swift programming language.<｜Assistant｜><think>\n"
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testDefaultTemplateFromArrayInConfig() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testTemplateFromArgumentWithEnum() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        // Purposely not using the correct template for this model to verify that the template from the config is not being used
        let mistral7BDefaultTemplate = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .literal(mistral7BDefaultTemplate))
        let encodedTarget = [1, 518, 25580, 29962, 20355, 915, 278, 14156, 8720, 4086, 29889, 518, 29914, 25580, 29962]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testTemplateFromArgumentWithString() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        // Purposely not using the correct template for this model to verify that the template from the config is not being used
        let mistral7BDefaultTemplate = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: mistral7BDefaultTemplate)
        let encodedTarget = [1, 518, 25580, 29962, 20355, 915, 278, 14156, 8720, 4086, 29889, 518, 29914, 25580, 29962]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testNamedTemplateFromArgument() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        // Normally it is not necessary to specify the name `default`, but I'm not aware of models with lists of templates in the config that are not `default` or `tool_use`
        let encoded = try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .name("default"))
        let encodedTarget = [1, 29473, 3, 28752, 1040, 4672, 2563, 17060, 4610, 29491, 29473, 4]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<s> [INST] Describe the Swift programming language. [/INST]"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    /// https://github.com/huggingface/transformers/pull/33957
    /// .jinja files have been introduced!
    func testJinjaOnlyTemplate() async throws {
        // Repo only contains .jinja file, no chat_template.json
        let tokenizer = try await AutoTokenizer.from(pretrained: "FL33TW00D-HF/jinja-test")
        let encoded = try tokenizer.applyChatTemplate(messages: messages)
        let encodedTarget = [151643, 151669, 74785, 279, 23670, 15473, 4128, 13, 151670]
        let decoded = tokenizer.decode(tokens: encoded)
        let decodedTarget = "<｜begin▁of▁sentence｜><｜User｜>Describe the Swift programming language.<｜Assistant｜>"
        XCTAssertEqual(encoded, encodedTarget)
        XCTAssertEqual(decoded, decodedTarget)
    }

    func testQwen2_5WithTools() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2.5-7B-Instruct-4bit")

        let weatherQueryMessages: [[String: String]] = [
            [
                "role": "user",
                "content": "What is the weather in Paris today?",
            ],
        ]

        let getCurrentWeatherToolSpec: [String: Any] = [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        ],
                        "unit": [
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        ],
                    ],
                    "required": ["location"],
                ],
            ],
        ]

        let encoded = try tokenizer.applyChatTemplate(messages: weatherQueryMessages, tools: [getCurrentWeatherToolSpec])
        let decoded = tokenizer.decode(tokens: encoded)

        func assertDictsAreEqual(_ actual: [String: Any], _ expected: [String: Any]) {
            for (key, value) in actual {
                if let nestedDict = value as? [String: Any], let nestedDict2 = expected[key] as? [String: Any] {
                    assertDictsAreEqual(nestedDict, nestedDict2)
                } else if let arrayValue = value as? [String] {
                    let expectedArrayValue = expected[key] as? [String]
                    XCTAssertNotNil(expectedArrayValue)
                    XCTAssertEqual(Set(arrayValue), Set(expectedArrayValue!))
                } else {
                    XCTAssertEqual(value as? String, expected[key] as? String)
                }
            }
        }

        if let startRange = decoded.range(of: "<tools>\n"),
           let endRange = decoded.range(of: "\n</tools>", range: startRange.upperBound..<decoded.endIndex)
        {
            let toolsSection = String(decoded[startRange.upperBound..<endRange.lowerBound])
            if let toolsDict = try? JSONSerialization.jsonObject(with: toolsSection.data(using: .utf8)!) as? [String: Any] {
                assertDictsAreEqual(toolsDict, getCurrentWeatherToolSpec)
            } else {
                XCTFail("Failed to decode tools section")
            }
        } else {
            XCTFail("Failed to find tools section")
        }

        let expectedPromptStart = """
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

        # Tools

        You may call one or more functions to assist with the user query.

        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        """

        let expectedPromptEnd = """
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call><|im_end|>
        <|im_start|>user
        What is the weather in Paris today?<|im_end|>
        <|im_start|>assistant

        """

        XCTAssertTrue(decoded.hasPrefix(expectedPromptStart), "Prompt should start with expected system message")
        XCTAssertTrue(decoded.hasSuffix(expectedPromptEnd), "Prompt should end with expected format")
    }

    func testHasChatTemplate() async throws {
        var tokenizer = try await AutoTokenizer.from(pretrained: "google-bert/bert-base-uncased")
        XCTAssertFalse(tokenizer.hasChatTemplate)

        tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        XCTAssertTrue(tokenizer.hasChatTemplate)
    }

    /// Test for vision models with a vision chat template in chat_template.json
    func testChatTemplateFromChatTemplateJson() async throws {
        let visionMessages = [
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
            ] as [String: Any],
        ] as [[String: Any]]
        // Qwen 2 VL does not have a chat_template.json file. The chat template is in tokenizer_config.json.
        let qwen2VLTokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2-VL-7B-Instruct-4bit")
        // Qwen 2.5 VL has a chat_template.json file with a different chat template than the one in tokenizer_config.json.
        let qwen2_5VLTokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        let qwen2VLEncoded = try qwen2VLTokenizer.applyChatTemplate(messages: visionMessages)
        let qwen2VLDecoded = qwen2VLTokenizer.decode(tokens: qwen2VLEncoded)
        let qwen2_5VLEncoded = try qwen2_5VLTokenizer.applyChatTemplate(messages: visionMessages)
        let qwen2_5VLDecoded = qwen2_5VLTokenizer.decode(tokens: qwen2_5VLEncoded)
        let expectedOutput = """
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What's in this image?<|vision_start|><|image_pad|><|vision_end|><|im_end|>
        <|im_start|>assistant

        """
        XCTAssertEqual(qwen2VLEncoded, qwen2_5VLEncoded, "Encoded sequences should be equal")
        XCTAssertEqual(qwen2VLDecoded, qwen2_5VLDecoded, "Decoded sequences should be equal")
        XCTAssertEqual(qwen2_5VLDecoded, expectedOutput, "Decoded sequence should match expected output")
    }

    func testApplyTemplateError() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "google-bert/bert-base-uncased")
        XCTAssertFalse(tokenizer.hasChatTemplate)
        XCTAssertThrowsError(try tokenizer.applyChatTemplate(messages: [])) { error in
            guard let tokenizerError = error as? TokenizerError else {
                XCTFail("Expected error of type TokenizerError, but got \(type(of: error))")
                return
            }
            if case .missingChatTemplate = tokenizerError {
                // Correct error caught, test passes
            } else {
                XCTFail("Expected .missingChatTemplate, but got \(tokenizerError)")
            }
        }
    }
}
