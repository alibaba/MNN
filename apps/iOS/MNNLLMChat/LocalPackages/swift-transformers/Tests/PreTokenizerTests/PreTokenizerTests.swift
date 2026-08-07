//
//  PreTokenizerTests.swift
//
//  Created by Jan Krukowski on 23/11/2023.
//

import Hub
@testable import Tokenizers
import XCTest

class PreTokenizerTests: XCTestCase {
    func testWhitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))

        XCTAssertEqual(
            preTokenizer.preTokenize(text: "Hey friend!"),
            ["Hey", "friend!"]
        )
        XCTAssertEqual(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", "friend!", "How", "are", "you?!?"]
        )
        XCTAssertEqual(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  "),
            ["Hey,", "friend,", "what's", "up?"]
        )
    }

    func testPunctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))

        XCTAssertEqual(
            preTokenizer.preTokenize(text: "Hey friend!"),
            ["Hey friend", "!"]
        )
        XCTAssertEqual(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey friend", "!", "     How are you", "?!?"]
        )
        XCTAssertEqual(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  "),
            ["   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  "]
        )
    }

    func testByteLevelPreTokenizer() {
        let preTokenizer1 = ByteLevelPreTokenizer(config: Config([String: Config]()))

        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!"),
            ["Hey", "Ġfriend", "!"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  "),
            ["ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ"]
        )

        let preTokenizer2 = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))

        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "Hey friend!"),
            ["ĠHey", "Ġfriend", "Ġ!"]
        )
        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?"),
            ["ĠHey", "Ġfriend", "Ġ!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "Ġ?!?"]
        )
        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  "),
            ["ĠĠ", "ĠHey", "Ġ,", "ĠĠĠ", "Ġfriend", "Ġ,", "ĠĠĠ", "Ġwhat", "Ġ's", "Ġup", "Ġ?", "ĠĠ"]
        )

        let preTokenizer3 = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "Hey friend!"),
            ["HeyĠfriend!"]
        )
        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?"),
            ["HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"]
        )
        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "   Hey,    friend,    what's up?  "),
            ["ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"]
        )
    }

    func testDigitsPreTokenizer() {
        let preTokenizer1 = DigitsPreTokenizer(config: Config([String: Config]()))

        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "1 12 123! 1234abc"),
            ["1", " ", "12", " ", "123", "! ", "1234", "abc"]
        )

        let preTokenizer2 = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "1 12 123! 1234abc"),
            ["1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc"]
        )
    }

    func testSplitPreTokenizer() {
        let preTokenizer1 = SplitPreTokenizer(config: Config(["pattern": ["String": " "]]))
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!"),
            ["Hey", " ", "friend!"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  "),
            [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "]
        )

        let preTokenizer2 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "\\s"]]))
        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "Hey friend!"),
            ["Hey", " ", "friend!"]
        )
        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"]
        )
        XCTAssertEqual(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  "),
            [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "]
        )

        let preTokenizer3 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"], "invert": true]))
        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "Hello"),
            ["Hello"]
        )

        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "Hey friend!"),
            ["Hey", " friend", "!"]
        )
        XCTAssertEqual(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", " friend", "!", "    ", " How", " are", " you", "?!?"]
        )
    }

    /// https://github.com/huggingface/tokenizers/pull/1357
    func testMetaspacePreTokenizer() {
        // Prepend "always"
        let preTokenizer = MetaspacePreTokenizer(config: Config([
            "add_prefix_space": true,
            "replacement": "▁",
            "prepend_scheme": "always",
        ]))

        // TODO: different sections on <s>
        let text = "Hey my friend <s>how▁are you"
        let tokens = text
            .split(by: "<s>", includeSeparators: true)
            .flatMap { preTokenizer.preTokenize(text: $0) }

        XCTAssertEqual(
            tokens,
            ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        )
    }

    func testBertPreTokenizer() {
        let preTokenizer1 = BertPreTokenizer(config: Config([String: Config]()))
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!"),
            ["Hey", "friend", "!"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?"),
            ["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,    what's up?  "),
            ["Hey", ",", "friend", ",", "what", "\'", "s", "up", "?"]
        )
        XCTAssertEqual(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,  0 99  what's up?  "),
            ["Hey", ",", "friend", ",", "0", "99", "what", "\'", "s", "up", "?"]
        )
    }
}
