//
//  BertTokenizerTests.swift
//  CoreMLBertTests
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

@testable import Hub
@testable import Tokenizers
import XCTest

class BertTokenizerTests: XCTestCase {
    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    lazy var bertTokenizer: BertTokenizer = {
        let vocab = {
            let url = Bundle.module.url(forResource: "bert-vocab", withExtension: "txt")!
            let vocabTxt = try! String(contentsOf: url)
            let tokens = vocabTxt.split(separator: "\n").map { String($0) }
            var vocab: [String: Int] = [:]
            for (i, token) in tokens.enumerated() {
                vocab[token] = i
            }
            return vocab
        }()

        return BertTokenizer(vocab: vocab, merges: nil)
    }()

    func testBasicTokenizer() {
        let basicTokenizer = BasicTokenizer()

        let text = "Brave gaillard, d'où [UNK] êtes vous?"
        let tokens = ["brave", "gaillard", ",", "d", "\'", "ou", "[UNK]", "etes", "vous", "?"]

        XCTAssertEqual(
            basicTokenizer.tokenize(text: text), tokens
        )
        // Verify that `XCTAssertEqual` does what deep equality checks on arrays of strings.
        XCTAssertEqual(["foo", "bar"], ["foo", "bar"])
    }

    /// For each Squad question tokenized by python, check that we get the same output through the `BasicTokenizer`
    func testFullBasicTokenizer() {
        let url = Bundle.module.url(forResource: "basic_tokenized_questions", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let sampleTokens = try! decoder.decode([[String]].self, from: json)

        let basicTokenizer = BasicTokenizer()

        XCTAssertEqual(sampleTokens.count, Squad.examples.count)

        for (i, example) in Squad.examples.enumerated() {
            let output = basicTokenizer.tokenize(text: example.question)
            XCTAssertEqual(output, sampleTokens[i])
        }
    }

    /// For each Squad question tokenized by python, check that we get the same output through the whole `BertTokenizer`
    func testFullBertTokenizer() {
        let url = Bundle.module.url(forResource: "tokenized_questions", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let sampleTokens = try! decoder.decode([[Int]].self, from: json)

        let tokenizer = bertTokenizer

        XCTAssertEqual(sampleTokens.count, Squad.examples.count)

        for (i, example) in Squad.examples.enumerated() {
            let output = tokenizer.tokenizeToIds(text: example.question)
            XCTAssertEqual(output, sampleTokens[i])
        }
    }

    func testMixedChineseEnglishTokenization() {
        let tokenizer = bertTokenizer
        let text = "你好，世界！Hello, world!"
        let expectedTokens = ["[UNK]", "[UNK]", "，", "世", "[UNK]", "！", "hello", ",", "world", "!"]
        let tokens = tokenizer.tokenize(text: text)

        XCTAssertEqual(tokens, expectedTokens)
    }

    func testPureChineseTokenization() {
        let tokenizer = bertTokenizer
        let text = "明日，大家上山看日出。"
        let expectedTokens = ["明", "日", "，", "大", "家", "上", "山", "[UNK]", "日", "出", "。"]
        let tokens = tokenizer.tokenize(text: text)

        XCTAssertEqual(tokens, expectedTokens)
    }

    func testChineseWithNumeralsTokenization() {
        let tokenizer = bertTokenizer
        let text = "2020年奥运会在东京举行。"
        let expectedTokens = ["2020", "年", "[UNK]", "[UNK]", "会", "[UNK]", "[UNK]", "京", "[UNK]", "行", "。"]
        let tokens = tokenizer.tokenize(text: text)

        XCTAssertEqual(tokens, expectedTokens)
    }

    func testChineseWithSpecialTokens() {
        let tokenizer = bertTokenizer
        let text = "[CLS] 机器学习是未来。 [SEP]"
        let expectedTokens = ["[CLS]", "[UNK]", "[UNK]", "学", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "。", "[SEP]"]
        let tokens = tokenizer.tokenize(text: text)

        XCTAssertEqual(tokens, expectedTokens)
    }

    func testPerformanceExample() {
        let tokenizer = bertTokenizer

        // This is an example of a performance test case.
        measure {
            // Put the code you want to measure the time of here.
            _ = tokenizer.tokenizeToIds(text: "Brave gaillard, d'où [UNK] êtes vous?")
        }
    }

    func testWordpieceDetokenizer() {
        struct QuestionTokens: Codable {
            let original: String
            let basic: [String]
            let wordpiece: [String]
        }

        let url = Bundle.module.url(forResource: "question_tokens", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let questionTokens = try! decoder.decode([QuestionTokens].self, from: json)
        let tokenizer = bertTokenizer

        for question in questionTokens {
            XCTAssertEqual(question.basic.joined(separator: " "), tokenizer.convertWordpieceToBasicTokenList(question.wordpiece))
        }
    }

    func testEncoderDecoder() {
        let text = """
        Wake up (Wake up)
        Grab a brush and put a little makeup
        Hide your scars to fade away the shakeup (Hide the scars to fade away the shakeup)
        Why'd you leave the keys upon the table?
        Here you go, create another fable, you wanted to
        Grab a brush and put a little makeup, you wanted to
        Hide the scars to fade away the shakeup, you wanted to
        Why'd you leave the keys upon the table? You wanted to
        """

        // Not sure if there's a way to achieve a non-destructive round-trip
        let decoded = """
        wake up ( wake up )
        grab a brush and put a little makeup
        hide your scars to fade away the shakeup ( hide the scars to fade away the shakeup )
        why \' d you leave the keys upon the table ?
        here you go , create another fable , you wanted to
        grab a brush and put a little makeup , you wanted to
        hide the scars to fade away the shakeup , you wanted to
        why \' d you leave the keys upon the table ? you wanted to
        """

        let tokenizer = bertTokenizer
        for (line, expected) in zip(text.split(separator: "\n"), decoded.split(separator: "\n")) {
            let encoded = tokenizer.encode(text: String(line))
            let decoded = tokenizer.decode(tokens: encoded)
            XCTAssertEqual(decoded, String(expected))
        }
    }

    func testBertTokenizerAddedTokensRecognized() async throws {
        let base: URL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appending(component: "huggingface-tests")
        let hubApi = HubApi(downloadBase: base)
        let configuration = LanguageModelConfigurationFromHub(modelName: "google-bert/bert-base-uncased", hubApi: hubApi)
        guard let tokenizerConfig = try await configuration.tokenizerConfig else { fatalError("missing tokenizer config") }
        let tokenizerData = try await configuration.tokenizerData
        let addedTokens = [
            "[ROAD]": 60_001,
            "[RIVER]": 60_002,
            "[BUILDING]": 60_003,
            "[PARK]": 60_004,
            "[BUFFER]": 60_005,
            "[INTERSECT]": 60_006,
            "[UNION]": 60_007,
        ]
        let tokenizer = try BertTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
        for (token, idx) in addedTokens {
            XCTAssertEqual(tokenizer.convertTokenToId(token), idx)
        }
        for (token, idx) in addedTokens {
            XCTAssertEqual(tokenizer.convertIdToToken(idx), token)
        }

        // Reading added_tokens from tokenizer.json
        XCTAssertEqual(tokenizer.convertTokenToId("[PAD]"), 0)
        XCTAssertEqual(tokenizer.convertTokenToId("[UNK]"), 100)
        XCTAssertEqual(tokenizer.convertTokenToId("[CLS]"), 101)
        XCTAssertEqual(tokenizer.convertTokenToId("[SEP]"), 102)
        XCTAssertEqual(tokenizer.convertTokenToId("[MASK]"), 103)
    }
}
