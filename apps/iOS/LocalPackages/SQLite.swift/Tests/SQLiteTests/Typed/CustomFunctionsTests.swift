import XCTest
import SQLite

// https://github.com/stephencelis/SQLite.swift/issues/1071
#if !os(Linux)

class CustomFunctionNoArgsTests: SQLiteTestCase {
    typealias FunctionNoOptional              = () -> Expression<String>
    typealias FunctionResultOptional          = () -> Expression<String?>

    func testFunctionNoOptional() throws {
        let _: FunctionNoOptional = try db.createFunction("test", deterministic: true) {
            "a"
        }
        let result = try db.prepare("SELECT test()").scalar() as! String
        XCTAssertEqual("a", result)
    }

    func testFunctionResultOptional() throws {
        let _: FunctionResultOptional = try db.createFunction("test", deterministic: true) {
            "a"
        }
        let result = try db.prepare("SELECT test()").scalar() as! String?
        XCTAssertEqual("a", result)
    }
}

class CustomFunctionWithOneArgTests: SQLiteTestCase {
    typealias FunctionNoOptional              = (Expression<String>) -> Expression<String>
    typealias FunctionLeftOptional            = (Expression<String?>) -> Expression<String>
    typealias FunctionResultOptional          = (Expression<String>) -> Expression<String?>
    typealias FunctionLeftResultOptional      = (Expression<String?>) -> Expression<String?>

    func testFunctionNoOptional() throws {
        let _: FunctionNoOptional = try db.createFunction("test", deterministic: true) { a in
            "b" + a
        }
        let result = try db.prepare("SELECT test(?)").scalar("a") as! String
        XCTAssertEqual("ba", result)
    }

    func testFunctionLeftOptional() throws {
        let _: FunctionLeftOptional = try db.createFunction("test", deterministic: true) { a in
            "b" + a!
        }
        let result = try db.prepare("SELECT test(?)").scalar("a") as! String
        XCTAssertEqual("ba", result)
    }

    func testFunctionResultOptional() throws {
        let _: FunctionResultOptional = try db.createFunction("test", deterministic: true) { a in
            "b" + a
        }
        let result = try db.prepare("SELECT test(?)").scalar("a") as! String
        XCTAssertEqual("ba", result)
    }

    func testFunctionLeftResultOptional() throws {
        let _: FunctionLeftResultOptional = try db.createFunction("test", deterministic: true) { (a: String?) -> String? in
            "b" + a!
        }
        let result = try db.prepare("SELECT test(?)").scalar("a") as! String
        XCTAssertEqual("ba", result)
    }
}

class CustomFunctionWithTwoArgsTests: SQLiteTestCase {
    typealias FunctionNoOptional              = (Expression<String>, Expression<String>) -> Expression<String>
    typealias FunctionLeftOptional            = (Expression<String?>, Expression<String>) -> Expression<String>
    typealias FunctionRightOptional           = (Expression<String>, Expression<String?>) -> Expression<String>
    typealias FunctionResultOptional          = (Expression<String>, Expression<String>) -> Expression<String?>
    typealias FunctionLeftRightOptional       = (Expression<String?>, Expression<String?>) -> Expression<String>
    typealias FunctionLeftResultOptional      = (Expression<String?>, Expression<String>) -> Expression<String?>
    typealias FunctionRightResultOptional     = (Expression<String>, Expression<String?>) -> Expression<String?>
    typealias FunctionLeftRightResultOptional = (Expression<String?>, Expression<String?>) -> Expression<String?>

    func testNoOptional() throws {
        let _: FunctionNoOptional = try db.createFunction("test", deterministic: true) { a, b in
            a + b
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String
        XCTAssertEqual("ab", result)
    }

    func testLeftOptional() throws {
        let _: FunctionLeftOptional = try db.createFunction("test", deterministic: true) { a, b in
            a! + b
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String
        XCTAssertEqual("ab", result)
    }

    func testRightOptional() throws {
        let _: FunctionRightOptional = try db.createFunction("test", deterministic: true) { a, b in
            a + b!
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String
        XCTAssertEqual("ab", result)
    }

    func testResultOptional() throws {
        let _: FunctionResultOptional = try db.createFunction("test", deterministic: true) { a, b in
            a + b
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String?
        XCTAssertEqual("ab", result)
    }

    func testFunctionLeftRightOptional() throws {
        let _: FunctionLeftRightOptional = try db.createFunction("test", deterministic: true) { a, b in
            a! + b!
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String
        XCTAssertEqual("ab", result)
    }

    func testFunctionLeftResultOptional() throws {
        let _: FunctionLeftResultOptional = try db.createFunction("test", deterministic: true) { a, b in
            a! + b
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String?
        XCTAssertEqual("ab", result)
    }

    func testFunctionRightResultOptional() throws {
        let _: FunctionRightResultOptional = try db.createFunction("test", deterministic: true) { a, b in
            a + b!
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String?
        XCTAssertEqual("ab", result)
    }

    func testFunctionLeftRightResultOptional() throws {
        let _: FunctionLeftRightResultOptional = try db.createFunction("test", deterministic: true) { a, b in
            a! + b!
        }
        let result = try db.prepare("SELECT test(?, ?)").scalar("a", "b") as! String?
        XCTAssertEqual("ab", result)
    }
}

class CustomFunctionTruncation: SQLiteTestCase {
    // https://github.com/stephencelis/SQLite.swift/issues/468
    func testStringTruncation() throws {
        _ = try db.createFunction("customLower") { (value: String) in value.lowercased() }
        let result = try db.prepare("SELECT customLower(?)").scalar("TÖL-AA 12") as? String
        XCTAssertEqual("töl-aa 12", result)
    }
}

#endif
