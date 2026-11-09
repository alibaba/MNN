import XCTest
@testable import SQLite

class CoreFunctionsTests: XCTestCase {

    func test_round_wrapsDoubleExpressionsWithRoundFunction() {
        assertSQL("round(\"double\")", double.round())
        assertSQL("round(\"doubleOptional\")", doubleOptional.round())

        assertSQL("round(\"double\", 1)", double.round(1))
        assertSQL("round(\"doubleOptional\", 2)", doubleOptional.round(2))
    }

    func test_random_generatesExpressionWithRandomFunction() {
        assertSQL("random()", Expression<Int64>.random())
        assertSQL("random()", Expression<Int>.random())
    }

    func test_length_wrapsStringExpressionWithLengthFunction() {
        assertSQL("length(\"string\")", string.length)
        assertSQL("length(\"stringOptional\")", stringOptional.length)
    }

    func test_lowercaseString_wrapsStringExpressionWithLowerFunction() {
        assertSQL("lower(\"string\")", string.lowercaseString)
        assertSQL("lower(\"stringOptional\")", stringOptional.lowercaseString)
    }

    func test_uppercaseString_wrapsStringExpressionWithUpperFunction() {
        assertSQL("upper(\"string\")", string.uppercaseString)
        assertSQL("upper(\"stringOptional\")", stringOptional.uppercaseString)
    }

    func test_like_buildsExpressionWithLikeOperator() {
        assertSQL("(\"string\" LIKE 'a%')", string.like("a%"))
        assertSQL("(\"stringOptional\" LIKE 'b%')", stringOptional.like("b%"))

        assertSQL("(\"string\" LIKE '%\\%' ESCAPE '\\')", string.like("%\\%", escape: "\\"))
        assertSQL("(\"stringOptional\" LIKE '_\\_' ESCAPE '\\')", stringOptional.like("_\\_", escape: "\\"))

        assertSQL("(\"string\" LIKE \"a\")", string.like(Expression<String>("a")))
        assertSQL("(\"stringOptional\" LIKE \"a\")", stringOptional.like(Expression<String>("a")))

        assertSQL("(\"string\" LIKE \"a\" ESCAPE '\\')", string.like(Expression<String>("a"), escape: "\\"))
        assertSQL("(\"stringOptional\" LIKE \"a\" ESCAPE '\\')", stringOptional.like(Expression<String>("a"), escape: "\\"))

        assertSQL("('string' LIKE \"a\")", "string".like(Expression<String>("a")))
        assertSQL("('string' LIKE \"a\" ESCAPE '\\')", "string".like(Expression<String>("a"), escape: "\\"))
    }

    func test_glob_buildsExpressionWithGlobOperator() {
        assertSQL("(\"string\" GLOB 'a*')", string.glob("a*"))
        assertSQL("(\"stringOptional\" GLOB 'b*')", stringOptional.glob("b*"))
    }

    func test_match_buildsExpressionWithMatchOperator() {
        assertSQL("(\"string\" MATCH 'a*')", string.match("a*"))
        assertSQL("(\"stringOptional\" MATCH 'b*')", stringOptional.match("b*"))
    }

    func test_regexp_buildsExpressionWithRegexpOperator() {
        assertSQL("(\"string\" REGEXP '^.+@.+\\.com$')", string.regexp("^.+@.+\\.com$"))
        assertSQL("(\"stringOptional\" REGEXP '^.+@.+\\.net$')", stringOptional.regexp("^.+@.+\\.net$"))
    }

    func test_collate_buildsExpressionWithCollateOperator() {
        assertSQL("(\"string\" COLLATE BINARY)", string.collate(.binary))
        assertSQL("(\"string\" COLLATE NOCASE)", string.collate(.nocase))
        assertSQL("(\"string\" COLLATE RTRIM)", string.collate(.rtrim))
        assertSQL("(\"string\" COLLATE \"CUSTOM\")", string.collate(.custom("CUSTOM")))

        assertSQL("(\"stringOptional\" COLLATE BINARY)", stringOptional.collate(.binary))
        assertSQL("(\"stringOptional\" COLLATE NOCASE)", stringOptional.collate(.nocase))
        assertSQL("(\"stringOptional\" COLLATE RTRIM)", stringOptional.collate(.rtrim))
        assertSQL("(\"stringOptional\" COLLATE \"CUSTOM\")", stringOptional.collate(.custom("CUSTOM")))
    }

    func test_ltrim_wrapsStringWithLtrimFunction() {
        assertSQL("ltrim(\"string\")", string.ltrim())
        assertSQL("ltrim(\"stringOptional\")", stringOptional.ltrim())

        assertSQL("ltrim(\"string\", ' ')", string.ltrim([" "]))
        assertSQL("ltrim(\"stringOptional\", ' ')", stringOptional.ltrim([" "]))
    }

    func test_ltrim_wrapsStringWithRtrimFunction() {
        assertSQL("rtrim(\"string\")", string.rtrim())
        assertSQL("rtrim(\"stringOptional\")", stringOptional.rtrim())

        assertSQL("rtrim(\"string\", ' ')", string.rtrim([" "]))
        assertSQL("rtrim(\"stringOptional\", ' ')", stringOptional.rtrim([" "]))
    }

    func test_ltrim_wrapsStringWithTrimFunction() {
        assertSQL("trim(\"string\")", string.trim())
        assertSQL("trim(\"stringOptional\")", stringOptional.trim())

        assertSQL("trim(\"string\", ' ')", string.trim([" "]))
        assertSQL("trim(\"stringOptional\", ' ')", stringOptional.trim([" "]))
    }

    func test_replace_wrapsStringWithReplaceFunction() {
        assertSQL("replace(\"string\", '@example.com', '@example.net')", string.replace("@example.com", with: "@example.net"))
        assertSQL("replace(\"stringOptional\", '@example.net', '@example.com')", stringOptional.replace("@example.net", with: "@example.com"))
    }

    func test_substring_wrapsStringWithSubstrFunction() {
        assertSQL("substr(\"string\", 1, 2)", string.substring(1, length: 2))
        assertSQL("substr(\"stringOptional\", 2, 1)", stringOptional.substring(2, length: 1))
    }

    func test_subscriptWithRange_wrapsStringWithSubstrFunction() {
        assertSQL("substr(\"string\", 1, 2)", string[1..<3])
        assertSQL("substr(\"stringOptional\", 2, 1)", stringOptional[2..<3])
    }

    func test_nilCoalescingOperator_wrapsOptionalsWithIfnullFunction() {
        assertSQL("ifnull(\"intOptional\", 1)", intOptional ?? 1)
        // AssertSQL("ifnull(\"doubleOptional\", 1.0)", doubleOptional ?? 1) // rdar://problem/21677256
        XCTAssertEqual("ifnull(\"doubleOptional\", 1.0)", (doubleOptional ?? 1).asSQL())
        assertSQL("ifnull(\"stringOptional\", 'literal')", stringOptional ?? "literal")

        assertSQL("ifnull(\"intOptional\", \"int\")", intOptional ?? int)
        assertSQL("ifnull(\"doubleOptional\", \"double\")", doubleOptional ?? double)
        assertSQL("ifnull(\"stringOptional\", \"string\")", stringOptional ?? string)

        assertSQL("ifnull(\"intOptional\", \"intOptional\")", intOptional ?? intOptional)
        assertSQL("ifnull(\"doubleOptional\", \"doubleOptional\")", doubleOptional ?? doubleOptional)
        assertSQL("ifnull(\"stringOptional\", \"stringOptional\")", stringOptional ?? stringOptional)
    }

    func test_absoluteValue_wrapsNumberWithAbsFucntion() {
        assertSQL("abs(\"int\")", int.absoluteValue)
        assertSQL("abs(\"intOptional\")", intOptional.absoluteValue)

        assertSQL("abs(\"double\")", double.absoluteValue)
        assertSQL("abs(\"doubleOptional\")", doubleOptional.absoluteValue)
    }

    func test_contains_buildsExpressionWithInOperator() {
        assertSQL("(\"string\" IN ('hello', 'world'))", ["hello", "world"].contains(string))
        assertSQL("(\"stringOptional\" IN ('hello', 'world'))", ["hello", "world"].contains(stringOptional))
    }

}
