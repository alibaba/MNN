package com.alibaba.mnnllm.android.chat.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentCalculatorTest {
    @Test
    fun `evaluates finite arithmetic percentages constants and functions`() {
        val expressions = mapOf(
            "2+3*4" to 14.0, "2^3^2" to 512.0, "15%*200" to 30.0, "10%3" to 1.0,
            "sqrt(16)+abs(-3)" to 7.0, "round(-2.5)" to -3.0, "floor(2.9)+ceil(1.1)" to 4.0,
            "ln(e)" to 1.0, "log(100)" to 2.0, "1e-3+2E2" to 200.001,
            "（1,234.5×2）=" to 2469.0, "\\frac{10}{4}" to 2.5,
        )
        for ((expression, value) in expressions) assertEquals(expression, value, AgentCalculator.evaluate(expression), 1e-9)
        assertTrue(AgentCalculator.observation("225.83*7.12").contains("result: 1607.9096"))
        assertEquals("0.000000000001", AgentCalculator.format(1e-12))
    }

    @Test
    fun `rejects code domain errors oversized expressions and every recursion path`() {
        val invalid = listOf("Runtime.exec('x')", "1/0", "sqrt(-1)", "ln(0)", "2;3", "x=2*3", "sin(1)",
            "1e999", "9^999", "-".repeat(100) + "1", "1^".repeat(100) + "1", "(".repeat(100) + "1" + ")".repeat(100),
            "1+".repeat(300) + "1", "1..2", "", "2+", "2*(3")
        for (expression in invalid) assertThrows(expression, IllegalArgumentException::class.java) { AgentCalculator.evaluate(expression) }
        assertFalse(AgentCalculator.observation("1/0").startsWith("[CALC_RESULT]"))
    }

    @Test
    fun `answer repair compatibility methods never rewrite prose arithmetic`() {
        val replies = listOf("-2*3=-6", "2^3+1=9", "124.44 × 7.2 = 8959.68", "2026-10-17 = 10",
            "  ```text\n-2*3=-6\n```  ", "unrelated text", "")
        for (reply in replies) {
            assertEquals(reply, AgentCalculator.correctArithmetic(reply))
            assertEquals(reply, AgentCalculator.repairArithmeticAnswers(reply))
        }
    }

    @Test
    fun `exponentiation binds above signs accepts negative powers and associates right`() {
        val expressions = mapOf("-2^2" to -4.0, "(-2)^2" to 4.0, "2^-2" to 0.25, "2^3^2" to 512.0,
            "-2^-2" to -0.25, "2^-2^2" to 0.0625, "-2^2*3" to -12.0, "(-2)^3" to -8.0,
            "2^(-2)" to 0.25, "+2^2" to 4.0)
        for ((expression, expected) in expressions) assertEquals(expression, expected, AgentCalculator.evaluate(expression), 1e-12)
    }
}
