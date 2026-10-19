//
//  AddedTokensTests.swift
//
//
//  Created by Pedro Cuenca on 20240426.
//

import Hub
import Tokenizers
import XCTest

class AddedTokensTests: XCTestCase {
    func testPhiAddedTokens() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        let inputIds = tokenizer("This is the <|end|>. My only friend, the <|end|>")
        XCTAssertEqual(inputIds, [910, 338, 278, 29871, 32007, 29889, 1619, 871, 5121, 29892, 278, 29871, 32007])

        let decoded = tokenizer.decode(tokens: inputIds)
        XCTAssertEqual(decoded, "This is the <|end|>. My only friend, the <|end|>")
    }

    func testGemmaAddedTokens() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/gemma-tokenizer")
        let inputIds = tokenizer("This\n\nis\na\ntest.")
        XCTAssertEqual(inputIds, [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265])

        let decoded = tokenizer.decode(tokens: inputIds)
        XCTAssertEqual(decoded, "<bos>This\n\nis\na\ntest.")
    }

    func testSplitWithCaptureGroups() {
        let addedTokensRegexp = #"(<\|end\|>)\s*|(<\|raw\|>)\s*"#
        let captureRegex = try! NSRegularExpression(pattern: addedTokensRegexp, options: [])

        XCTAssertEqual(
            "eating <|raw|> meat <|end|> That's all".split(by: captureRegex),
            ["eating ", "<|raw|>", "meat ", "<|end|>", "That's all"]
        )

        XCTAssertEqual(
            "<|raw|>".split(by: captureRegex),
            ["<|raw|>"]
        )

        XCTAssertEqual(
            "This string doesn't have those separators".split(by: captureRegex),
            ["This string doesn't have those separators"]
        )

        XCTAssertEqual(
            "start <|end|>".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|> ".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|>       ".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|>       for real".split(by: captureRegex),
            ["start ", "<|end|>", "for real"]
        )

        XCTAssertEqual(
            "<|raw|><|end|>".split(by: captureRegex),
            ["<|raw|>", "<|end|>"]
        )
    }
}
