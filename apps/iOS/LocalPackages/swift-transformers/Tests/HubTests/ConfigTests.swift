//
//  ConfigTests.swift
//  swift-transformers
//
//  Created by Piotr Kowalczuk on 13.03.25.
//

import Foundation
import Jinja
import XCTest

@testable import Hub

class ConfigGeneralTests: XCTestCase {
    func testHashable() throws {
        let testCases: [(Config.Data, Config.Data)] = [
            (Config.Data.integer(1), Config.Data.integer(2)),
            (Config.Data.string("a"), Config.Data.string("2")),
            (Config.Data.boolean(true), Config.Data.string("T")),
            (Config.Data.boolean(true), Config.Data.boolean(false)),
            (Config.Data.floating(1.1), Config.Data.floating(1.1000001)),
            (Config.Data.token((1, "a")), Config.Data.token((1, "b"))),
            (Config.Data.token((1, "a")), Config.Data.token((2, "a"))),
            (Config.Data.dictionary(["1": Config()]), Config.Data.dictionary(["1": 1])),
            (Config.Data.dictionary(["1": 10]), Config.Data.dictionary(["2": 10])),
            (Config.Data.array(["1", "2"]), Config.Data.array(["1", "3"])),
            (Config.Data.array([1, 2]), Config.Data.array([2, 1])),
            (Config.Data.array([true, false]), Config.Data.array([true, true])),
        ]

        for (lhs, rhs) in testCases {
            var lhsh = Hasher()
            var rhsh = Hasher()

            lhs.hash(into: &lhsh)
            rhs.hash(into: &rhsh)

            XCTAssertNotEqual(lhsh.finalize(), rhsh.finalize())
        }
    }
}

class ConfigAsLiteralTests: XCTestCase {
    func testStringLiteral() throws {
        let cfg: Config = "test"
        XCTAssertEqual(cfg, "test")
    }

    func testIntegerLiteral() throws {
        let cfg: Config = 678
        XCTAssertEqual(cfg, 678)
    }

    func testBooleanLiteral() throws {
        let cfg: Config = true
        XCTAssertEqual(cfg, true)
    }

    func testFloatLiteral() throws {
        let cfg: Config = 1.1
        XCTAssertEqual(cfg, 1.1)
    }

    func testDictionaryLiteral() throws {
        let cfg: Config = ["key": 1.1]
        XCTAssertEqual(cfg["key"].floating(or: 0), 1.1)
    }

    func testArrayLiteral() throws {
        let cfg: Config = [1.1, 1.2]
        XCTAssertEqual(cfg[0], 1.1)
        XCTAssertEqual(cfg[1], 1.2)
    }
}

class ConfigAccessorsTests: XCTestCase {
    func testKeySubscript() throws {
        let cfg: Config = ["key": 1.1]

        XCTAssertEqual(cfg["key"], 1.1)
        XCTAssertTrue(cfg["non_existent"].isNull())
        XCTAssertTrue(cfg[1].isNull())
    }

    func testIndexSubscript() throws {
        let cfg: Config = [1, 2, 3, 4]

        XCTAssertEqual(cfg[1], 2)
        XCTAssertTrue(cfg[99].isNull())
        XCTAssertTrue(cfg[-1].isNull())
    }

    func testDynamicLookup() throws {
        let cfg: Config = ["model_type": "bert"]

        XCTAssertEqual(cfg["model_type"], "bert")
        XCTAssertEqual(cfg.modelType, "bert")
        XCTAssertEqual(cfg.model_type, "bert")
        XCTAssertTrue(cfg.unknown_key.isNull())
    }

    func testArray() throws {
        let cfg: Config = [1, 2, 3, 4]

        XCTAssertEqual(cfg.array(), [1, 2, 3, 4])
        XCTAssertEqual(cfg.get(), [1, 2, 3, 4])
        XCTAssertEqual(cfg.get(or: []), [1, 2, 3, 4])
        XCTAssertTrue(cfg["fake_key"].isNull())
        XCTAssertNil(cfg.dictionary())
        XCTAssertEqual(cfg.dictionary(or: ["a": 1]), ["a": 1])
    }

    func testArrayOfStrings() throws {
        let cfg: Config = ["a", "b", "c"]

        XCTAssertEqual(cfg.array(), ["a", "b", "c"])
        XCTAssertEqual(cfg.get(), ["a", "b", "c"])
        XCTAssertEqual(cfg.get(), [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        XCTAssertEqual(cfg.get(or: []), [BinaryDistinctString("a"), BinaryDistinctString("b"), BinaryDistinctString("c")])
        XCTAssertEqual(cfg.get(or: []), ["a", "b", "c"])
        XCTAssertNil(cfg.dictionary())
        XCTAssertEqual(cfg.dictionary(or: ["a": 1]), ["a": 1])
    }

    func testArrayOfConfigs() throws {
        let cfg: Config = [Config("a"), Config("b")]

        XCTAssertEqual(cfg.array(), ["a", "b"])
        XCTAssertEqual(cfg.get(), ["a", "b"])
        XCTAssertEqual(cfg.get(), [BinaryDistinctString("a"), BinaryDistinctString("b")])
        XCTAssertEqual(cfg.get(or: []), [BinaryDistinctString("a"), BinaryDistinctString("b")])
        XCTAssertEqual(cfg.get(or: []), ["a", "b"])
        XCTAssertNil(cfg.dictionary())
        XCTAssertEqual(cfg.dictionary(or: ["a": 1]), ["a": 1])
    }

    func testDictionary() throws {
        let cfg: Config = ["a": 1, "b": 2, "c": 3, "d": 4]

        XCTAssertEqual(cfg.dictionary(), ["a": 1, "b": 2, "c": 3, "d": 4])
        XCTAssertEqual(cfg.get(), ["a": 1, "b": 2, "c": 3, "d": 4])
        XCTAssertEqual(cfg.get(or: [:]), ["a": 1, "b": 2, "c": 3, "d": 4])
        XCTAssertTrue(cfg[666].isNull())
        XCTAssertNil(cfg.array())
        XCTAssertEqual(cfg.array(or: ["a"]), ["a"])
    }

    func testDictionaryOfConfigs() throws {
        let cfg: Config = ["a": .init([1, 2]), "b": .init([3, 4])]
        let exp = [BinaryDistinctString("a"): Config([1, 2]), BinaryDistinctString("b"): Config([3, 4])]

        XCTAssertEqual(cfg.dictionary(), exp)
        XCTAssertEqual(cfg.get(), exp)
        XCTAssertEqual(cfg.get(or: [:]), exp)
        XCTAssertTrue(cfg[666].isNull())
        XCTAssertNil(cfg.array())
        XCTAssertEqual(cfg.array(or: ["a"]), ["a"])
    }
}

class ConfigCodableTests: XCTestCase {
    func testCompleteHappyExample() throws {
        let cfg: Config = [
            "dict_of_floats": ["key1": 1.1],
            "dict_of_ints": ["key2": 100],
            "dict_of_strings": ["key3": "abc"],
            "dict_of_bools": ["key4": false],
            "dict_of_dicts": ["key5": ["key_inside": 99]],
            "dict_of_tokens": ["key6": .init((12, "dfe"))],
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["a", "b"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [["key7": 1.1], ["key8": 1.2]],
            "arr_of_tokens": [.init((1, "a")), .init((2, "b"))],
            "int": 678,
            "float": 1.1,
            "string": "test",
            "bool": true,
            "token": .init((1, "test")),
            "null": Config(),
        ]

        let data = try JSONEncoder().encode(cfg)
        let got = try JSONDecoder().decode(Config.self, from: data)

        XCTAssertEqual(got, cfg)
        XCTAssertEqual(got["dict_of_floats"]["key1"], 1.1)
        XCTAssertEqual(got["dict_of_ints"]["key2"], 100)
        XCTAssertEqual(got["dict_of_strings"]["key3"], "abc")
        XCTAssertEqual(got["dict_of_bools"]["key4"], false)
        XCTAssertEqual(got["dict_of_dicts"]["key5"]["key_inside"], 99)
        XCTAssertEqual(got["dict_of_tokens"]["key6"].token()?.0, 12)
        XCTAssertEqual(got["dict_of_tokens"]["key6"].token()?.1, "dfe")
        XCTAssertEqual(got["arr_empty"].array()?.count, 0)
        XCTAssertEqual(got["arr_of_ints"], [1, 2, 3])
        XCTAssertEqual(got["arr_of_floats"], [1.1, 1.2])
        XCTAssertEqual(got["arr_of_strings"], ["a", "b"])
        XCTAssertEqual(got["arr_of_bools"], [true, false])
        XCTAssertEqual(got["arr_of_dicts"][1]["key8"], 1.2)
        XCTAssert(got["arr_of_tokens"][1].token(or: (0, "")) == (2, "b"))
        XCTAssertNil(got["arr_of_tokens"][2].token())
        XCTAssertEqual(got["int"], 678)
        XCTAssertEqual(got["float"], 1.1)
        XCTAssertEqual(got["string"], "test")
        XCTAssertEqual(got["bool"], true)
        XCTAssert(got["token"].token(or: (0, "")) == (1, "test"))
        XCTAssertTrue(got["null"].isNull())
    }
}

class ConfigEquatableTests: XCTestCase {
    func testString() throws {
        let cfg = Config("a")

        XCTAssertEqual(cfg, "a")
        XCTAssertEqual(cfg.get(), "a")
        XCTAssertEqual(cfg.get(or: "b"), "a")
        XCTAssertEqual(cfg.string(), "a")
        XCTAssertEqual(cfg.string(or: "b"), "a")
        XCTAssertEqual(cfg.get(), BinaryDistinctString("a"))
        XCTAssertEqual(cfg.get(or: "b"), BinaryDistinctString("a"))
        XCTAssertEqual(cfg.binaryDistinctString(), "a")
        XCTAssertEqual(cfg.binaryDistinctString(or: "b"), "a")
    }

    func testInteger() throws {
        let cfg = Config(1)

        XCTAssertEqual(cfg, 1)
        XCTAssertEqual(cfg.get(), 1)
        XCTAssertEqual(cfg.get(or: 2), 1)
        XCTAssertEqual(cfg.integer(), 1)
        XCTAssertEqual(cfg.integer(or: 2), 1)
    }

    func testFloating() throws {
        let testCases: [(Config, Float)] = [
            (Config(1.1), 1.1),
            (Config(1), 1.0),
        ]

        for (cfg, exp) in testCases {
            XCTAssertEqual(cfg, .init(exp))
            XCTAssertEqual(cfg.get(), exp)
            XCTAssertEqual(cfg.get(or: 2.2), exp)
            XCTAssertEqual(cfg.floating(), exp)
            XCTAssertEqual(cfg.floating(or: 2.2), exp)
        }
    }

    func testBoolean() throws {
        let testCases: [(Config, Bool)] = [
            (Config(true), true),
            (Config(1), true),
            (Config("T"), true),
            (Config("t"), true),
            (Config("TRUE"), true),
            (Config("True"), true),
            (Config("true"), true),
            (Config("F"), false),
            (Config("f"), false),
            (Config("FALSE"), false),
            (Config("False"), false),
            (Config("false"), false),
        ]

        for (cfg, exp) in testCases {
            XCTAssertEqual(cfg.get(), exp)
            XCTAssertEqual(cfg.get(or: !exp), exp)
            XCTAssertEqual(cfg.boolean(), exp)
            XCTAssertEqual(cfg.boolean(or: !exp), exp)
        }
    }

    func testToken() throws {
        let cfg = Config((1, "a"))
        let exp: (UInt, String) = (1, "a")

        XCTAssertEqual(cfg, .init((1, "a")))
        XCTAssert(cfg.get()! == exp)
        XCTAssert(cfg.get(or: (2, "b")) == exp)
        XCTAssert(cfg.token()! == exp)
        XCTAssert(cfg.token(or: (2, "b")) == exp)
    }

    func testDictionary() throws {
        let testCases: [(Config, Int)] = [
            (Config(["a": 1]), 1),
            (Config(["a": 2] as [NSString: Any]), 2),
            (Config(["a": 3] as [NSString: Config]), 3),
            (Config([BinaryDistinctString("a"): 4] as [BinaryDistinctString: Config]), 4),
            (Config(["a": Config(5)]), 5),
            (Config(["a": 6]), 6),
            (Config((BinaryDistinctString("a"), 7)), 7),
        ]

        for (cfg, exp) in testCases {
            XCTAssertEqual(cfg["a"], Config(exp))
            XCTAssertEqual(cfg.get(or: [:])["a"], Config(exp))
        }
    }
}

class ConfigTextEncodingTests: XCTestCase {
    private func createFile(with content: String, encoding: String.Encoding, fileName: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(fileName)
        guard let data = content.data(using: encoding) else {
            throw NSError(domain: "EncodingError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Could not encode string with \(encoding)"])
        }
        try data.write(to: fileURL)
        return fileURL
    }

    func testUtf16() throws {
        let json = """
            {
              "a": ["val_1", "val_2"],
              "b": 2,
              "c": [[10, "tkn_1"], [12, "tkn_2"], [4, "tkn_3"]],
              "d": false,
              "e": {
                "e_1": 1.1,
                "e_2": [1, 2, 3]
              },
              "f": null
            }
        """

        let urlUTF8 = try createFile(with: json, encoding: .utf8, fileName: "config_utf8.json")
        let urlUTF16LE = try createFile(with: json, encoding: .utf16LittleEndian, fileName: "config_utf16_le.json")
        let urlUTF16BE = try createFile(with: json, encoding: .utf16BigEndian, fileName: "config_utf16_be.json")

        let dataUTF8 = try Data(contentsOf: urlUTF8)
        let dataUTF16LE = try Data(contentsOf: urlUTF16LE)
        let dataUTF16BE = try Data(contentsOf: urlUTF16BE)

        XCTAssertNotEqual(dataUTF8.count, dataUTF16LE.count)
        XCTAssertNotEqual(dataUTF8.count, dataUTF16BE.count)

        let decoder = JSONDecoder()
        let configUTF8 = try decoder.decode(Config.self, from: dataUTF8)
        let configUTF16LE = try decoder.decode(Config.self, from: dataUTF16LE)
        let configUTF16BE = try decoder.decode(Config.self, from: dataUTF16BE)

        XCTAssertEqual(configUTF8, configUTF16LE)
        XCTAssertEqual(configUTF8, configUTF16BE)

        try FileManager.default.removeItem(at: urlUTF8)
        try FileManager.default.removeItem(at: urlUTF16LE)
        try FileManager.default.removeItem(at: urlUTF16BE)
    }

    func testUnicode() {
        // These are two different characters
        let json = "{\"vocab\": {\"à\": 1, \"à\": 2}}"
        let data = json.data(using: .utf8)
        let dict = try! JSONSerialization.jsonObject(with: data!, options: []) as! [NSString: Any]
        let config = Config(dict)

        let vocab = config["vocab"].dictionary(or: [:])

        XCTAssertEqual(vocab.count, 2)
    }
}

class ConfigTemplatingTests: XCTestCase {
    func testCompleteHappyExample() throws {
        let cfg = Config([
            "dict_of_floats": ["key1": 1.1],
            "dict_of_tokens": ["key6": .init((12, "dfe"))],
            "arr_empty": [],
            "arr_of_ints": [1, 2, 3],
            "arr_of_floats": [1.1, 1.2],
            "arr_of_strings": ["tre", "jeq"],
            "arr_of_bools": [true, false],
            "arr_of_dicts": [["key7": 1.1], ["key8": 1.2]],
            "arr_of_tokens": [.init((1, "ghz")), .init((2, "pkr"))],
            "int": 678,
            "float": 1.1,
            "string": "hha",
            "bool": true,
            "token": .init((1, "iop")),
            "null": Config(),
        ])
        let template = """
        {{ config["dict_of_floats"]["key1"] }}
        {{ config["dict_of_tokens"]["key6"]["12"] }}
        {{ config["arr_of_ints"][0] }}
        {{ config["arr_of_ints"][1] }}
        {{ config["arr_of_ints"][2] }}
        {{ config["arr_of_floats"][0] }}
        {{ config["arr_of_floats"][1] }}
        {{ config["arr_of_strings"][0] }}
        {{ config["arr_of_strings"][1] }}
        {{ config["arr_of_bools"][0] }}
        {{ config["arr_of_bools"][1] }}
        {{ config["arr_of_dicts"][0]["key7"] }}
        {{ config["arr_of_dicts"][1]["key8"] }}
        {{ config["arr_of_tokens"][0]["1"] }}
        {{ config["arr_of_tokens"][1]["2"] }}
        {{ config["int"] }}
        {{ config["float"] }}
        {{ config["string"] }}
        {{ config["bool"] }}
        {{ config["token"]["1"] }}
        """
        let exp = """
        1.1
        dfe
        1
        2
        3
        1.1
        1.2
        tre
        jeq
        true
        false
        1.1
        1.2
        ghz
        pkr
        678
        1.1
        hha
        true
        iop
        """

        let got = try Template(template).render([
            "config": cfg.toJinjaCompatible(),
        ])

        XCTAssertEqual(got, exp)
    }
}
