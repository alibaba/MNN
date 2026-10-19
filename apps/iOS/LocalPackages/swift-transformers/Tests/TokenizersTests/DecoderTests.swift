//
//  DecoderTests.swift
//
//  Created by Pedro Cuenca on 20231123.
//

import Hub
@testable import Tokenizers
import XCTest

class DecoderTests: XCTestCase {
    /// https://github.com/huggingface/tokenizers/pull/1357
    func testMetaspaceDecoder() {
        let decoder = MetaspaceDecoder(config: Config([
            "add_prefix_space": true,
            "replacement": "▁",
        ]))

        let tokens = ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        let decoded = decoder.decode(tokens: tokens)

        XCTAssertEqual(
            decoded,
            ["Hey", " my", " friend", " ", " <s>", " how", " are", " you"]
        )
    }

    func testWordPieceDecoder() {
        let config = Config(["prefix": "##", "cleanup": true])
        let decoder = WordPieceDecoder(config: config)

        let testCases: [([String], String)] = [
            (["##inter", "##national", "##ization"], "##internationalization"),
            (["##auto", "##mat", "##ic", "transmission"], "##automatic transmission"),
            (["who", "do", "##n't", "does", "n't", "can't"], "who don't doesn't can't"),
            (["##un", "##believ", "##able", "##fa", "##ntastic"], "##unbelievablefantastic"),
            (["this", "is", "un", "##believ", "##able", "fa", "##ntastic"], "this is unbelievable fantastic"),
            (["The", "##quick", "##brown", "fox"], "Thequickbrown fox"),
        ]

        for (tokens, expected) in testCases {
            let output = decoder.decode(tokens: tokens)
            XCTAssertEqual(output.joined(), expected)
        }
    }
}
