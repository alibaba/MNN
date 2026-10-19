import XCTest
import SQLite

class OperatorsTests: XCTestCase {

    func test_stringExpressionPlusStringExpression_buildsConcatenatingStringExpression() {
        assertSQL("(\"string\" || \"string\")", string + string)
        assertSQL("(\"string\" || \"stringOptional\")", string + stringOptional)
        assertSQL("(\"stringOptional\" || \"string\")", stringOptional + string)
        assertSQL("(\"stringOptional\" || \"stringOptional\")", stringOptional + stringOptional)
        assertSQL("(\"string\" || 'literal')", string + "literal")
        assertSQL("(\"stringOptional\" || 'literal')", stringOptional + "literal")
        assertSQL("('literal' || \"string\")", "literal" + string)
        assertSQL("('literal' || \"stringOptional\")", "literal" + stringOptional)
    }

    func test_numberExpression_plusNumberExpression_buildsAdditiveNumberExpression() {
        assertSQL("(\"int\" + \"int\")", int + int)
        assertSQL("(\"int\" + \"intOptional\")", int + intOptional)
        assertSQL("(\"intOptional\" + \"int\")", intOptional + int)
        assertSQL("(\"intOptional\" + \"intOptional\")", intOptional + intOptional)
        assertSQL("(\"int\" + 1)", int + 1)
        assertSQL("(\"intOptional\" + 1)", intOptional + 1)
        assertSQL("(1 + \"int\")", 1 + int)
        assertSQL("(1 + \"intOptional\")", 1 + intOptional)

        assertSQL("(\"double\" + \"double\")", double + double)
        assertSQL("(\"double\" + \"doubleOptional\")", double + doubleOptional)
        assertSQL("(\"doubleOptional\" + \"double\")", doubleOptional + double)
        assertSQL("(\"doubleOptional\" + \"doubleOptional\")", doubleOptional + doubleOptional)
        assertSQL("(\"double\" + 1.0)", double + 1)
        assertSQL("(\"doubleOptional\" + 1.0)", doubleOptional + 1)
        assertSQL("(1.0 + \"double\")", 1 + double)
        assertSQL("(1.0 + \"doubleOptional\")", 1 + doubleOptional)
    }

    func test_numberExpression_minusNumberExpression_buildsSubtractiveNumberExpression() {
        assertSQL("(\"int\" - \"int\")", int - int)
        assertSQL("(\"int\" - \"intOptional\")", int - intOptional)
        assertSQL("(\"intOptional\" - \"int\")", intOptional - int)
        assertSQL("(\"intOptional\" - \"intOptional\")", intOptional - intOptional)
        assertSQL("(\"int\" - 1)", int - 1)
        assertSQL("(\"intOptional\" - 1)", intOptional - 1)
        assertSQL("(1 - \"int\")", 1 - int)
        assertSQL("(1 - \"intOptional\")", 1 - intOptional)

        assertSQL("(\"double\" - \"double\")", double - double)
        assertSQL("(\"double\" - \"doubleOptional\")", double - doubleOptional)
        assertSQL("(\"doubleOptional\" - \"double\")", doubleOptional - double)
        assertSQL("(\"doubleOptional\" - \"doubleOptional\")", doubleOptional - doubleOptional)
        assertSQL("(\"double\" - 1.0)", double - 1)
        assertSQL("(\"doubleOptional\" - 1.0)", doubleOptional - 1)
        assertSQL("(1.0 - \"double\")", 1 - double)
        assertSQL("(1.0 - \"doubleOptional\")", 1 - doubleOptional)
    }

    func test_numberExpression_timesNumberExpression_buildsMultiplicativeNumberExpression() {
        assertSQL("(\"int\" * \"int\")", int * int)
        assertSQL("(\"int\" * \"intOptional\")", int * intOptional)
        assertSQL("(\"intOptional\" * \"int\")", intOptional * int)
        assertSQL("(\"intOptional\" * \"intOptional\")", intOptional * intOptional)
        assertSQL("(\"int\" * 1)", int * 1)
        assertSQL("(\"intOptional\" * 1)", intOptional * 1)
        assertSQL("(1 * \"int\")", 1 * int)
        assertSQL("(1 * \"intOptional\")", 1 * intOptional)

        assertSQL("(\"double\" * \"double\")", double * double)
        assertSQL("(\"double\" * \"doubleOptional\")", double * doubleOptional)
        assertSQL("(\"doubleOptional\" * \"double\")", doubleOptional * double)
        assertSQL("(\"doubleOptional\" * \"doubleOptional\")", doubleOptional * doubleOptional)
        assertSQL("(\"double\" * 1.0)", double * 1)
        assertSQL("(\"doubleOptional\" * 1.0)", doubleOptional * 1)
        assertSQL("(1.0 * \"double\")", 1 * double)
        assertSQL("(1.0 * \"doubleOptional\")", 1 * doubleOptional)
    }

    func test_numberExpression_dividedByNumberExpression_buildsDivisiveNumberExpression() {
        assertSQL("(\"int\" / \"int\")", int / int)
        assertSQL("(\"int\" / \"intOptional\")", int / intOptional)
        assertSQL("(\"intOptional\" / \"int\")", intOptional / int)
        assertSQL("(\"intOptional\" / \"intOptional\")", intOptional / intOptional)
        assertSQL("(\"int\" / 1)", int / 1)
        assertSQL("(\"intOptional\" / 1)", intOptional / 1)
        assertSQL("(1 / \"int\")", 1 / int)
        assertSQL("(1 / \"intOptional\")", 1 / intOptional)

        assertSQL("(\"double\" / \"double\")", double / double)
        assertSQL("(\"double\" / \"doubleOptional\")", double / doubleOptional)
        assertSQL("(\"doubleOptional\" / \"double\")", doubleOptional / double)
        assertSQL("(\"doubleOptional\" / \"doubleOptional\")", doubleOptional / doubleOptional)
        assertSQL("(\"double\" / 1.0)", double / 1)
        assertSQL("(\"doubleOptional\" / 1.0)", doubleOptional / 1)
        assertSQL("(1.0 / \"double\")", 1 / double)
        assertSQL("(1.0 / \"doubleOptional\")", 1 / doubleOptional)
    }

    func test_numberExpression_prefixedWithMinus_buildsInvertedNumberExpression() {
        assertSQL("-(\"int\")", -int)
        assertSQL("-(\"intOptional\")", -intOptional)

        assertSQL("-(\"double\")", -double)
        assertSQL("-(\"doubleOptional\")", -doubleOptional)
    }

    func test_integerExpression_moduloIntegerExpression_buildsModuloIntegerExpression() {
        assertSQL("(\"int\" % \"int\")", int % int)
        assertSQL("(\"int\" % \"intOptional\")", int % intOptional)
        assertSQL("(\"intOptional\" % \"int\")", intOptional % int)
        assertSQL("(\"intOptional\" % \"intOptional\")", intOptional % intOptional)
        assertSQL("(\"int\" % 1)", int % 1)
        assertSQL("(\"intOptional\" % 1)", intOptional % 1)
        assertSQL("(1 % \"int\")", 1 % int)
        assertSQL("(1 % \"intOptional\")", 1 % intOptional)
    }

    func test_integerExpression_bitShiftLeftIntegerExpression_buildsLeftShiftedIntegerExpression() {
        assertSQL("(\"int\" << \"int\")", int << int)
        assertSQL("(\"int\" << \"intOptional\")", int << intOptional)
        assertSQL("(\"intOptional\" << \"int\")", intOptional << int)
        assertSQL("(\"intOptional\" << \"intOptional\")", intOptional << intOptional)
        assertSQL("(\"int\" << 1)", int << 1)
        assertSQL("(\"intOptional\" << 1)", intOptional << 1)
        assertSQL("(1 << \"int\")", 1 << int)
        assertSQL("(1 << \"intOptional\")", 1 << intOptional)
    }

    func test_integerExpression_bitShiftRightIntegerExpression_buildsRightShiftedIntegerExpression() {
        assertSQL("(\"int\" >> \"int\")", int >> int)
        assertSQL("(\"int\" >> \"intOptional\")", int >> intOptional)
        assertSQL("(\"intOptional\" >> \"int\")", intOptional >> int)
        assertSQL("(\"intOptional\" >> \"intOptional\")", intOptional >> intOptional)
        assertSQL("(\"int\" >> 1)", int >> 1)
        assertSQL("(\"intOptional\" >> 1)", intOptional >> 1)
        assertSQL("(1 >> \"int\")", 1 >> int)
        assertSQL("(1 >> \"intOptional\")", 1 >> intOptional)
    }

    func test_integerExpression_bitwiseAndIntegerExpression_buildsAndedIntegerExpression() {
        assertSQL("(\"int\" & \"int\")", int & int)
        assertSQL("(\"int\" & \"intOptional\")", int & intOptional)
        assertSQL("(\"intOptional\" & \"int\")", intOptional & int)
        assertSQL("(\"intOptional\" & \"intOptional\")", intOptional & intOptional)
        assertSQL("(\"int\" & 1)", int & 1)
        assertSQL("(\"intOptional\" & 1)", intOptional & 1)
        assertSQL("(1 & \"int\")", 1 & int)
        assertSQL("(1 & \"intOptional\")", 1 & intOptional)
    }

    func test_integerExpression_bitwiseOrIntegerExpression_buildsOredIntegerExpression() {
        assertSQL("(\"int\" | \"int\")", int | int)
        assertSQL("(\"int\" | \"intOptional\")", int | intOptional)
        assertSQL("(\"intOptional\" | \"int\")", intOptional | int)
        assertSQL("(\"intOptional\" | \"intOptional\")", intOptional | intOptional)
        assertSQL("(\"int\" | 1)", int | 1)
        assertSQL("(\"intOptional\" | 1)", intOptional | 1)
        assertSQL("(1 | \"int\")", 1 | int)
        assertSQL("(1 | \"intOptional\")", 1 | intOptional)
    }

    func test_integerExpression_bitwiseExclusiveOrIntegerExpression_buildsOredIntegerExpression() {
        assertSQL("(~((\"int\" & \"int\")) & (\"int\" | \"int\"))", int ^ int)
        assertSQL("(~((\"int\" & \"intOptional\")) & (\"int\" | \"intOptional\"))", int ^ intOptional)
        assertSQL("(~((\"intOptional\" & \"int\")) & (\"intOptional\" | \"int\"))", intOptional ^ int)
        assertSQL("(~((\"intOptional\" & \"intOptional\")) & (\"intOptional\" | \"intOptional\"))", intOptional ^ intOptional)
        assertSQL("(~((\"int\" & 1)) & (\"int\" | 1))", int ^ 1)
        assertSQL("(~((\"intOptional\" & 1)) & (\"intOptional\" | 1))", intOptional ^ 1)
        assertSQL("(~((1 & \"int\")) & (1 | \"int\"))", 1 ^ int)
        assertSQL("(~((1 & \"intOptional\")) & (1 | \"intOptional\"))", 1 ^ intOptional)
    }

    func test_bitwiseNot_integerExpression_buildsComplementIntegerExpression() {
        assertSQL("~(\"int\")", ~int)
        assertSQL("~(\"intOptional\")", ~intOptional)
    }

    func test_equalityOperator_withEquatableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" = \"bool\")", bool == bool)
        assertSQL("(\"bool\" = \"boolOptional\")", bool == boolOptional)
        assertSQL("(\"boolOptional\" = \"bool\")", boolOptional == bool)
        assertSQL("(\"boolOptional\" = \"boolOptional\")", boolOptional == boolOptional)
        assertSQL("(\"bool\" = 1)", bool == true)
        assertSQL("(\"boolOptional\" = 1)", boolOptional == true)
        assertSQL("(1 = \"bool\")", true == bool)
        assertSQL("(1 = \"boolOptional\")", true == boolOptional)

        assertSQL("(\"boolOptional\" IS NULL)", boolOptional == nil)
        assertSQL("(NULL IS \"boolOptional\")", nil == boolOptional)
    }

    func test_isOperator_withEquatableExpressions_buildsBooleanExpression() {
       assertSQL("(\"bool\" IS \"bool\")", bool === bool)
       assertSQL("(\"bool\" IS \"boolOptional\")", bool === boolOptional)
       assertSQL("(\"boolOptional\" IS \"bool\")", boolOptional === bool)
       assertSQL("(\"boolOptional\" IS \"boolOptional\")", boolOptional === boolOptional)
       assertSQL("(\"bool\" IS 1)", bool === true)
       assertSQL("(\"boolOptional\" IS 1)", boolOptional === true)
       assertSQL("(1 IS \"bool\")", true === bool)
       assertSQL("(1 IS \"boolOptional\")", true === boolOptional)

       assertSQL("(\"boolOptional\" IS NULL)", boolOptional === nil)
       assertSQL("(NULL IS \"boolOptional\")", nil === boolOptional)
    }

    func test_isNotOperator_withEquatableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" IS NOT \"bool\")", bool !== bool)
        assertSQL("(\"bool\" IS NOT \"boolOptional\")", bool !== boolOptional)
        assertSQL("(\"boolOptional\" IS NOT \"bool\")", boolOptional !== bool)
        assertSQL("(\"boolOptional\" IS NOT \"boolOptional\")", boolOptional !== boolOptional)
        assertSQL("(\"bool\" IS NOT 1)", bool !== true)
        assertSQL("(\"boolOptional\" IS NOT 1)", boolOptional !== true)
        assertSQL("(1 IS NOT \"bool\")", true !== bool)
        assertSQL("(1 IS NOT \"boolOptional\")", true !== boolOptional)

        assertSQL("(\"boolOptional\" IS NOT NULL)", boolOptional !== nil)
        assertSQL("(NULL IS NOT \"boolOptional\")", nil !== boolOptional)
     }

    func test_inequalityOperator_withEquatableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" != \"bool\")", bool != bool)
        assertSQL("(\"bool\" != \"boolOptional\")", bool != boolOptional)
        assertSQL("(\"boolOptional\" != \"bool\")", boolOptional != bool)
        assertSQL("(\"boolOptional\" != \"boolOptional\")", boolOptional != boolOptional)
        assertSQL("(\"bool\" != 1)", bool != true)
        assertSQL("(\"boolOptional\" != 1)", boolOptional != true)
        assertSQL("(1 != \"bool\")", true != bool)
        assertSQL("(1 != \"boolOptional\")", true != boolOptional)

        assertSQL("(\"boolOptional\" IS NOT NULL)", boolOptional != nil)
        assertSQL("(NULL IS NOT \"boolOptional\")", nil != boolOptional)
    }

    func test_greaterThanOperator_withComparableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" > \"bool\")", bool > bool)
        assertSQL("(\"bool\" > \"boolOptional\")", bool > boolOptional)
        assertSQL("(\"boolOptional\" > \"bool\")", boolOptional > bool)
        assertSQL("(\"boolOptional\" > \"boolOptional\")", boolOptional > boolOptional)
        assertSQL("(\"bool\" > 1)", bool > true)
        assertSQL("(\"boolOptional\" > 1)", boolOptional > true)
        assertSQL("(1 > \"bool\")", true > bool)
        assertSQL("(1 > \"boolOptional\")", true > boolOptional)
    }

    func test_greaterThanOrEqualToOperator_withComparableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" >= \"bool\")", bool >= bool)
        assertSQL("(\"bool\" >= \"boolOptional\")", bool >= boolOptional)
        assertSQL("(\"boolOptional\" >= \"bool\")", boolOptional >= bool)
        assertSQL("(\"boolOptional\" >= \"boolOptional\")", boolOptional >= boolOptional)
        assertSQL("(\"bool\" >= 1)", bool >= true)
        assertSQL("(\"boolOptional\" >= 1)", boolOptional >= true)
        assertSQL("(1 >= \"bool\")", true >= bool)
        assertSQL("(1 >= \"boolOptional\")", true >= boolOptional)
    }

    func test_lessThanOperator_withComparableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" < \"bool\")", bool < bool)
        assertSQL("(\"bool\" < \"boolOptional\")", bool < boolOptional)
        assertSQL("(\"boolOptional\" < \"bool\")", boolOptional < bool)
        assertSQL("(\"boolOptional\" < \"boolOptional\")", boolOptional < boolOptional)
        assertSQL("(\"bool\" < 1)", bool < true)
        assertSQL("(\"boolOptional\" < 1)", boolOptional < true)
        assertSQL("(1 < \"bool\")", true < bool)
        assertSQL("(1 < \"boolOptional\")", true < boolOptional)
    }

    func test_lessThanOrEqualToOperator_withComparableExpressions_buildsBooleanExpression() {
        assertSQL("(\"bool\" <= \"bool\")", bool <= bool)
        assertSQL("(\"bool\" <= \"boolOptional\")", bool <= boolOptional)
        assertSQL("(\"boolOptional\" <= \"bool\")", boolOptional <= bool)
        assertSQL("(\"boolOptional\" <= \"boolOptional\")", boolOptional <= boolOptional)
        assertSQL("(\"bool\" <= 1)", bool <= true)
        assertSQL("(\"boolOptional\" <= 1)", boolOptional <= true)
        assertSQL("(1 <= \"bool\")", true <= bool)
        assertSQL("(1 <= \"boolOptional\")", true <= boolOptional)
    }

    func test_patternMatchingOperator_withComparableCountableClosedRange_buildsBetweenBooleanExpression() {
        assertSQL("\"int\" BETWEEN 0 AND 5", 0...5 ~= int)
        assertSQL("\"intOptional\" BETWEEN 0 AND 5", 0...5 ~= intOptional)
    }

    func test_patternMatchingOperator_withComparableClosedRange_buildsBetweenBooleanExpression() {
        assertSQL("\"double\" BETWEEN 1.2 AND 4.5", 1.2...4.5 ~= double)
        assertSQL("\"doubleOptional\" BETWEEN 1.2 AND 4.5", 1.2...4.5 ~= doubleOptional)
    }

    func test_patternMatchingOperator_withComparableRange_buildsBooleanExpression() {
        assertSQL("\"double\" >= 1.2 AND \"double\" < 4.5", 1.2..<4.5 ~= double)
        assertSQL("\"doubleOptional\" >= 1.2 AND \"doubleOptional\" < 4.5", 1.2..<4.5 ~= doubleOptional)
    }

    func test_patternMatchingOperator_withComparablePartialRangeThrough_buildsBooleanExpression() {
        assertSQL("\"double\" <= 4.5", ...4.5 ~= double)
        assertSQL("\"doubleOptional\" <= 4.5", ...4.5 ~= doubleOptional)
    }

    func test_patternMatchingOperator_withComparablePartialRangeUpTo_buildsBooleanExpression() {
        assertSQL("\"double\" < 4.5", ..<4.5 ~= double)
        assertSQL("\"doubleOptional\" < 4.5", ..<4.5 ~= doubleOptional)
    }

    func test_patternMatchingOperator_withComparablePartialRangeFrom_buildsBooleanExpression() {
        assertSQL("\"double\" >= 4.5", 4.5... ~= double)
        assertSQL("\"doubleOptional\" >= 4.5", 4.5... ~= doubleOptional)
    }

    func test_patternMatchingOperator_withComparableClosedRangeString_buildsBetweenBooleanExpression() {
        assertSQL("\"string\" BETWEEN 'a' AND 'b'", "a"..."b" ~= string)
        assertSQL("\"stringOptional\" BETWEEN 'a' AND 'b'", "a"..."b" ~= stringOptional)
    }

    func test_doubleAndOperator_withBooleanExpressions_buildsCompoundExpression() {
        assertSQL("(\"bool\" AND \"bool\")", bool && bool)
        assertSQL("(\"bool\" AND \"boolOptional\")", bool && boolOptional)
        assertSQL("(\"boolOptional\" AND \"bool\")", boolOptional && bool)
        assertSQL("(\"boolOptional\" AND \"boolOptional\")", boolOptional && boolOptional)
        assertSQL("(\"bool\" AND 1)", bool && true)
        assertSQL("(\"boolOptional\" AND 1)", boolOptional && true)
        assertSQL("(1 AND \"bool\")", true && bool)
        assertSQL("(1 AND \"boolOptional\")", true && boolOptional)
    }

    func test_andFunction_withBooleanExpressions_buildsCompoundExpression() {
        assertSQL("(\"bool\" AND \"bool\" AND \"bool\")", and([bool, bool, bool]))
        assertSQL("(\"bool\" AND \"bool\")", and([bool, bool]))
        assertSQL("(\"bool\")", and([bool]))

        assertSQL("(\"bool\" AND \"bool\" AND \"bool\")", and(bool, bool, bool))
        assertSQL("(\"bool\" AND \"bool\")", and(bool, bool))
        assertSQL("(\"bool\")", and(bool))
    }

    func test_doubleOrOperator_withBooleanExpressions_buildsCompoundExpression() {
        assertSQL("(\"bool\" OR \"bool\")", bool || bool)
        assertSQL("(\"bool\" OR \"boolOptional\")", bool || boolOptional)
        assertSQL("(\"boolOptional\" OR \"bool\")", boolOptional || bool)
        assertSQL("(\"boolOptional\" OR \"boolOptional\")", boolOptional || boolOptional)
        assertSQL("(\"bool\" OR 1)", bool || true)
        assertSQL("(\"boolOptional\" OR 1)", boolOptional || true)
        assertSQL("(1 OR \"bool\")", true || bool)
        assertSQL("(1 OR \"boolOptional\")", true || boolOptional)
    }

    func test_orFunction_withBooleanExpressions_buildsCompoundExpression() {
        assertSQL("(\"bool\" OR \"bool\" OR \"bool\")", or([bool, bool, bool]))
        assertSQL("(\"bool\" OR \"bool\")", or([bool, bool]))
        assertSQL("(\"bool\")", or([bool]))

        assertSQL("(\"bool\" OR \"bool\" OR \"bool\")", or(bool, bool, bool))
        assertSQL("(\"bool\" OR \"bool\")", or(bool, bool))
        assertSQL("(\"bool\")", or(bool))
    }

    func test_unaryNotOperator_withBooleanExpressions_buildsNotExpression() {
        assertSQL("NOT (\"bool\")", !bool)
        assertSQL("NOT (\"boolOptional\")", !boolOptional)
    }

    func test_precedencePreserved() {
        let n = Expression<Int>(value: 1)
        assertSQL("(((1 = 1) AND (1 = 1)) OR (1 = 1))", (n == n && n == n) || n == n)
        assertSQL("((1 = 1) AND ((1 = 1) OR (1 = 1)))", n == n && (n == n || n == n))
    }

    func test_dateExpressionLessGreater() {
        let begin = Date(timeIntervalSince1970: 0)
        assertSQL("(\"date\" < '1970-01-01T00:00:00.000')", date < begin)
        assertSQL("(\"date\" > '1970-01-01T00:00:00.000')", date > begin)
        assertSQL("(\"date\" >= '1970-01-01T00:00:00.000')", date >= begin)
        assertSQL("(\"date\" <= '1970-01-01T00:00:00.000')", date <= begin)
    }

    func test_dateExpressionRange() {
        let begin = Date(timeIntervalSince1970: 0)
        let end = Date(timeIntervalSince1970: 5000)
        assertSQL(
            "\"date\" >= '1970-01-01T00:00:00.000' AND \"date\" < '1970-01-01T01:23:20.000'",
            (begin..<end) ~= date
        )
    }

    func test_dateExpressionClosedRange() {
        let begin = Date(timeIntervalSince1970: 0)
        let end = Date(timeIntervalSince1970: 5000)
        assertSQL(
            "\"date\" BETWEEN '1970-01-01T00:00:00.000' AND '1970-01-01T01:23:20.000'",
            (begin...end) ~= date
        )
    }

}
