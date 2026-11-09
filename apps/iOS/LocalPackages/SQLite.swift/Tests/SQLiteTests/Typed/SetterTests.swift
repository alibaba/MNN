import XCTest
import SQLite

class SetterTests: XCTestCase {

    func test_setterAssignmentOperator_buildsSetter() {
        assertSQL("\"int\" = \"int\"", int <- int)
        assertSQL("\"int\" = 1", int <- 1)
        assertSQL("\"intOptional\" = \"int\"", intOptional <- int)
        assertSQL("\"intOptional\" = \"intOptional\"", intOptional <- intOptional)
        assertSQL("\"intOptional\" = 1", intOptional <- 1)
        assertSQL("\"intOptional\" = NULL", intOptional <- nil)
    }

    func test_plusEquals_withStringExpression_buildsSetter() {
        assertSQL("\"string\" = (\"string\" || \"string\")", string += string)
        assertSQL("\"string\" = (\"string\" || 'literal')", string += "literal")
        assertSQL("\"stringOptional\" = (\"stringOptional\" || \"string\")", stringOptional += string)
        assertSQL("\"stringOptional\" = (\"stringOptional\" || \"stringOptional\")", stringOptional += stringOptional)
        assertSQL("\"stringOptional\" = (\"stringOptional\" || 'literal')", stringOptional += "literal")
    }

    func test_plusEquals_withNumberExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" + \"int\")", int += int)
        assertSQL("\"int\" = (\"int\" + 1)", int += 1)
        assertSQL("\"intOptional\" = (\"intOptional\" + \"int\")", intOptional += int)
        assertSQL("\"intOptional\" = (\"intOptional\" + \"intOptional\")", intOptional += intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" + 1)", intOptional += 1)

        assertSQL("\"double\" = (\"double\" + \"double\")", double += double)
        assertSQL("\"double\" = (\"double\" + 1.0)", double += 1)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" + \"double\")", doubleOptional += double)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" + \"doubleOptional\")", doubleOptional += doubleOptional)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" + 1.0)", doubleOptional += 1)
    }

    func test_minusEquals_withNumberExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" - \"int\")", int -= int)
        assertSQL("\"int\" = (\"int\" - 1)", int -= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" - \"int\")", intOptional -= int)
        assertSQL("\"intOptional\" = (\"intOptional\" - \"intOptional\")", intOptional -= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" - 1)", intOptional -= 1)

        assertSQL("\"double\" = (\"double\" - \"double\")", double -= double)
        assertSQL("\"double\" = (\"double\" - 1.0)", double -= 1)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" - \"double\")", doubleOptional -= double)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" - \"doubleOptional\")", doubleOptional -= doubleOptional)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" - 1.0)", doubleOptional -= 1)
    }

    func test_timesEquals_withNumberExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" * \"int\")", int *= int)
        assertSQL("\"int\" = (\"int\" * 1)", int *= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" * \"int\")", intOptional *= int)
        assertSQL("\"intOptional\" = (\"intOptional\" * \"intOptional\")", intOptional *= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" * 1)", intOptional *= 1)

        assertSQL("\"double\" = (\"double\" * \"double\")", double *= double)
        assertSQL("\"double\" = (\"double\" * 1.0)", double *= 1)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" * \"double\")", doubleOptional *= double)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" * \"doubleOptional\")", doubleOptional *= doubleOptional)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" * 1.0)", doubleOptional *= 1)
    }

    func test_dividedByEquals_withNumberExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" / \"int\")", int /= int)
        assertSQL("\"int\" = (\"int\" / 1)", int /= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" / \"int\")", intOptional /= int)
        assertSQL("\"intOptional\" = (\"intOptional\" / \"intOptional\")", intOptional /= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" / 1)", intOptional /= 1)

        assertSQL("\"double\" = (\"double\" / \"double\")", double /= double)
        assertSQL("\"double\" = (\"double\" / 1.0)", double /= 1)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" / \"double\")", doubleOptional /= double)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" / \"doubleOptional\")", doubleOptional /= doubleOptional)
        assertSQL("\"doubleOptional\" = (\"doubleOptional\" / 1.0)", doubleOptional /= 1)
    }

    func test_moduloEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" % \"int\")", int %= int)
        assertSQL("\"int\" = (\"int\" % 1)", int %= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" % \"int\")", intOptional %= int)
        assertSQL("\"intOptional\" = (\"intOptional\" % \"intOptional\")", intOptional %= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" % 1)", intOptional %= 1)
    }

    func test_leftShiftEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" << \"int\")", int <<= int)
        assertSQL("\"int\" = (\"int\" << 1)", int <<= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" << \"int\")", intOptional <<= int)
        assertSQL("\"intOptional\" = (\"intOptional\" << \"intOptional\")", intOptional <<= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" << 1)", intOptional <<= 1)
    }

    func test_rightShiftEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" >> \"int\")", int >>= int)
        assertSQL("\"int\" = (\"int\" >> 1)", int >>= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" >> \"int\")", intOptional >>= int)
        assertSQL("\"intOptional\" = (\"intOptional\" >> \"intOptional\")", intOptional >>= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" >> 1)", intOptional >>= 1)
    }

    func test_bitwiseAndEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" & \"int\")", int &= int)
        assertSQL("\"int\" = (\"int\" & 1)", int &= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" & \"int\")", intOptional &= int)
        assertSQL("\"intOptional\" = (\"intOptional\" & \"intOptional\")", intOptional &= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" & 1)", intOptional &= 1)
    }

    func test_bitwiseOrEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (\"int\" | \"int\")", int |= int)
        assertSQL("\"int\" = (\"int\" | 1)", int |= 1)
        assertSQL("\"intOptional\" = (\"intOptional\" | \"int\")", intOptional |= int)
        assertSQL("\"intOptional\" = (\"intOptional\" | \"intOptional\")", intOptional |= intOptional)
        assertSQL("\"intOptional\" = (\"intOptional\" | 1)", intOptional |= 1)
    }

    func test_bitwiseExclusiveOrEquals_withIntegerExpression_buildsSetter() {
        assertSQL("\"int\" = (~((\"int\" & \"int\")) & (\"int\" | \"int\"))", int ^= int)
        assertSQL("\"int\" = (~((\"int\" & 1)) & (\"int\" | 1))", int ^= 1)
        assertSQL("\"intOptional\" = (~((\"intOptional\" & \"int\")) & (\"intOptional\" | \"int\"))", intOptional ^= int)
        assertSQL("\"intOptional\" = (~((\"intOptional\" & \"intOptional\")) & (\"intOptional\" | \"intOptional\"))", intOptional ^= intOptional)
        assertSQL("\"intOptional\" = (~((\"intOptional\" & 1)) & (\"intOptional\" | 1))", intOptional ^= 1)
    }

    func test_postfixPlus_withIntegerValue_buildsSetter() {
        assertSQL("\"int\" = (\"int\" + 1)", int++)
        assertSQL("\"intOptional\" = (\"intOptional\" + 1)", intOptional++)
    }

    func test_postfixMinus_withIntegerValue_buildsSetter() {
        assertSQL("\"int\" = (\"int\" - 1)", int--)
        assertSQL("\"intOptional\" = (\"intOptional\" - 1)", intOptional--)
    }

}
