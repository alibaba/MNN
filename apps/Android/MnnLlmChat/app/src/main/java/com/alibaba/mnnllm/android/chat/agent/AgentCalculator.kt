package com.alibaba.mnnllm.android.chat.agent

import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import java.util.Locale
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

object AgentCalculator {
    const val MAX_LENGTH = 512
    private const val MAX_DEPTH = 32

    fun evaluate(expression: String): Double {
        require(expression.length in 1..MAX_LENGTH) { "Empty or oversized expression" }
        val normalized = normalize(expression)
        require(normalized.length in 1..MAX_LENGTH) { "Invalid expression" }
        return Parser(normalized).parse()
    }

    fun normalize(expression: String): String {
        var text = expression
        for ((from, to) in listOf("\\times" to "*", "\\cdot" to "*", "\\div" to "/", "\\left" to "",
            "\\right" to "", "\\(" to "", "\\)" to "", "\\*" to "*", "×" to "*", "✕" to "*",
            "÷" to "/", "（" to "(", "）" to ")", "−" to "-", "－" to "-", "＋" to "+", "**" to "^", "π" to "pi")) {
            text = text.replace(from, to)
        }
        // ICU（Android）要求转义字面 }，JVM 单测环境不会暴露此问题
        text = text.replace(Regex("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}"), "(($1)/($2))")
            .replace(Regex("(?<=\\d),(?=\\d{3}\\b)"), "").trim()
        if (text.endsWith("=")) text = text.dropLast(1).trim()
        return text
    }

    fun format(value: Double): String {
        require(value.isFinite()) { "Non-finite result" }
        if (value == 0.0) return "0"
        return BigDecimal.valueOf(value).round(MathContext(15, RoundingMode.HALF_UP)).stripTrailingZeros().toPlainString()
    }

    fun observation(expression: String): String = try {
        "[CALC_RESULT]\nexpression: ${normalize(expression)}\nresult: ${format(evaluate(expression))}"
    } catch (e: IllegalArgumentException) {
        "[CALC_ERROR] ${e.message ?: "Invalid arithmetic"}. Use plain arithmetic only."
    }

    fun correctArithmetic(text: String): String = text

    fun repairArithmeticAnswers(text: String): String = text

    private class Parser(private val text: String) {
        private var index = 0
        private var depth = 0

        fun parse(): Double {
            val value = expression()
            require(peek() == null) { "Unexpected token" }
            return finite(value)
        }

        private fun finite(value: Double): Double {
            require(value.isFinite()) { "Non-finite result or division by zero" }
            return value
        }

        private fun peek(): Char? {
            while (index < text.length && text[index].isWhitespace()) index++
            return text.getOrNull(index)
        }

        private fun expression(): Double {
            var value = term()
            while (peek() == '+' || peek() == '-') {
                val operator = text[index++]
                val rhs = term()
                value = finite(if (operator == '+') value + rhs else value - rhs)
            }
            return value
        }

        private fun term(): Double {
            var value = unary()
            while (peek() in listOf('*', '/', '%')) {
                val operator = text[index++]
                val rhs = unary()
                value = finite(when (operator) { '*' -> value * rhs; '/' -> value / rhs; else -> value % rhs })
            }
            return value
        }

        private fun nested(block: () -> Double): Double {
            require(++depth <= MAX_DEPTH) { "Expression too deeply nested" }
            return try { block() } finally { depth-- }
        }

        private fun power(): Double = nested {
            val base = primary()
            if (peek() == '^') { index++; finite(base.pow(unary())) } else base
        }

        private fun unary(): Double = nested {
            when (peek()) {
                '-' -> { index++; -unary() }
                '+' -> { index++; unary() }
                else -> power()
            }
        }

        private fun primary(): Double {
            val ch = peek() ?: throw IllegalArgumentException("Expression ends too early")
            if (ch == '(') {
                index++
                val value = nested { expression() }
                require(peek() == ')') { "Missing closing parenthesis" }
                index++
                return percent(value)
            }
            if (ch in '0'..'9' || ch == '.') {
                val start = index
                while (index < text.length && (text[index] in '0'..'9' || text[index] == '.')) index++
                if (text.getOrNull(index) in listOf('e', 'E') &&
                    text.getOrNull(index + 1)?.let { it in '0'..'9' || it in "+-" } == true) {
                    index++
                    if (text.getOrNull(index) in listOf('+', '-')) index++
                    while (index < text.length && text[index] in '0'..'9') index++
                }
                val number = text.substring(start, index).toDoubleOrNull()
                    ?: throw IllegalArgumentException("Invalid number")
                return percent(finite(number))
            }
            if (ch in 'a'..'z' || ch in 'A'..'Z') {
                val start = index
                while (index < text.length && text[index].isLetter()) index++
                val name = text.substring(start, index).lowercase(Locale.ROOT)
                if (name == "pi") return percent(Math.PI)
                if (name == "e") return percent(Math.E)
                require(name in setOf("sqrt", "abs", "round", "floor", "ceil", "ln", "log")) { "Unknown function" }
                require(peek() == '(') { "Missing function argument" }
                index++
                val argument = nested { expression() }
                require(peek() == ')') { "Missing closing parenthesis" }
                index++
                val value = when (name) {
                    "sqrt" -> sqrt(argument)
                    "abs" -> abs(argument)
                    "round" -> if (argument >= 0) floor(argument + 0.5) else ceil(argument - 0.5)
                    "floor" -> floor(argument)
                    "ceil" -> ceil(argument)
                    "ln" -> ln(argument)
                    else -> log10(argument)
                }
                return percent(finite(value))
            }
            throw IllegalArgumentException("Unexpected token")
        }

        private fun percent(value: Double): Double {
            if (peek() != '%') return value
            var look = index + 1
            while (look < text.length && text[look].isWhitespace()) look++
            val next = text.getOrNull(look)
            if (next != null && (next in '0'..'9' || next == '(' || next == '.' || next.isLetter())) return value
            index++
            return value / 100
        }
    }
}
