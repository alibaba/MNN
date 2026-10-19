//
//  CoreTagTests.swift
//  Jinja
//
//  Created by Anthony DePasquale on 07.01.2025.
//

// Adapted from https://github.com/pallets/jinja/blob/main/tests/test_core_tags.py

import XCTest
@testable import Jinja

final class IfConditionTests: XCTestCase {
    // MARK: - If Condition Tests

    func testSimpleIf() throws {
        let template = try Template("{% if true %}...{% endif %}")
        let result = try template.render([:])
        XCTAssertEqual(result, "...")
    }

    func testIfElif() throws {
        let template = try Template(
            """
            {% if false %}XXX{% elif true %}...{% else %}XXX{% endif %}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result, "...")
    }

    func testIfElse() throws {
        let template = try Template("{% if false %}XXX{% else %}...{% endif %}")
        let result = try template.render([:])
        XCTAssertEqual(result, "...")
    }

    func testEmptyIf() throws {
        let template = try Template("[{% if true %}{% else %}{% endif %}]")
        let result = try template.render([:])
        XCTAssertEqual(result, "[]")
    }

    // TODO: Make this test pass
    //    func testCompleteIf() throws {
    //        let template = try Template(
    //            """
    //            {% if a %}A{% elif b %}B{% elif c == d %}C{% else %}D{% endif %}
    //            """
    //        )
    //        let result = try template.render([
    //            "a": 0,
    //            "b": false,
    //            "c": 42,
    //            "d": 42.0,
    //        ])
    //        XCTAssertEqual(result, "C")
    //    }

    // MARK: - Set Tests

    func testNormalSet() throws {
        let template = try Template("{% set foo = 1 %}{{ foo }}")
        let result = try template.render([:])
        XCTAssertEqual(result, "1")
    }

    // TODO: Make this test pass
    //    func testBlockSet() throws {
    //        let template = try Template("{% set foo %}42{% endset %}{{ foo }}")
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "42")
    //    }

    func testNamespace() throws {
        let template = try Template(
            """
            {% set ns = namespace() %}{% set ns.bar = '42' %}{{ ns.bar }}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result, "42")
    }

    // TODO: Make this test pass
    //    func testNamespaceLoop() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace(found=false) %}\
    //            {% for x in range(4) %}\
    //            {% if x == v %}\
    //            {% set ns.found = true %}\
    //            {% endif %}\
    //            {% endfor %}\
    //            {{ ns.found }}
    //            """
    //        )
    //
    //        let result1 = try template.render(["v": 3])
    //        XCTAssertEqual(result1, "true")
    //
    //        let result2 = try template.render(["v": 4])
    //        XCTAssertEqual(result2, "false")
    //    }
}

final class ForLoopTests: XCTestCase {
    // MARK: - For Loop Tests

    func testSimpleForLoop() throws {
        let template = try Template("{% for item in seq %}{{ item }}{% endfor %}")
        let result = try template.render(["seq": Array(0 ... 9)])
        XCTAssertEqual(result, "0123456789")
    }

    // TODO: Make this test pass
    //    func testForLoopWithElse() throws {
    //        let template = try Template("{% for item in seq %}XXX{% else %}...{% endfor %}")
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "...")
    //    }

    func testForLoopElseScopingItem() throws {
        let template = try Template("{% for item in [] %}{% else %}{{ item }}{% endfor %}")
        let result = try template.render(["item": 42])
        XCTAssertEqual(result, "42")
    }

    // TODO: Make this test pass
    //    func testEmptyBlocks() throws {
    //        let template = try Template("<{% for item in seq %}{% else %}{% endfor %}>")
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "<>")
    //    }

    func testContextVars() throws {
        let template = try Template(
            """
            {% for item in seq -%}
            {{ loop.index }}|{{ loop.index0 }}|{{ loop.revindex }}|{{
                loop.revindex0 }}|{{ loop.first }}|{{ loop.last }}|{{
               loop.length }}###{% endfor %}
            """
        )

        let result = try template.render(["seq": [42, 24]])
        let parts = result.split(separator: "###")
        XCTAssertEqual(parts.count, 2)

        let one = String(parts[0]).split(separator: "|")
        let two = String(parts[1]).split(separator: "|")

        // First iteration checks
        XCTAssertEqual(one[0], "1")  // index
        XCTAssertEqual(one[1], "0")  // index0
        XCTAssertEqual(one[2], "2")  // revindex
        XCTAssertEqual(one[3], "1")  // revindex0
        XCTAssertEqual(one[4], "true")  // first
        XCTAssertEqual(one[5], "false")  // last
        XCTAssertEqual(one[6], "2")  // length

        // Second iteration checks
        XCTAssertEqual(two[0], "2")  // index
        XCTAssertEqual(two[1], "1")  // index0
        XCTAssertEqual(two[2], "1")  // revindex
        XCTAssertEqual(two[3], "0")  // revindex0
        XCTAssertEqual(two[4], "false")  // first
        XCTAssertEqual(two[5], "true")  // last
        XCTAssertEqual(two[6], "2")  // length
    }

    // TODO: Make this test pass
    //    func testCycling() throws {
    //        let template = try Template(
    //            """
    //            {% for item in seq %}{{ loop.cycle('<1>', '<2>') }}{% endfor %}\
    //            {% for item in seq %}{{ loop.cycle(*through) }}{% endfor %}
    //            """
    //        )
    //        let result = try template.render([
    //            "seq": Array(0 ... 3),
    //            "through": ["<1>", "<2>"],
    //        ])
    //        XCTAssertEqual(result, "<1><2><1><2><1><2><1><2>")
    //    }

    func testLookaround() throws {
        let template = try Template(
            """
            {% for item in seq -%}
            {{ loop.previtem|default('x') }}-{{ item }}-{{ loop.nextitem|default('x') }}|
            {%- endfor %}
            """
        )
        let result = try template.render(["seq": Array(0 ... 3)])
        XCTAssertEqual(result, "x-0-1|0-1-2|1-2-3|2-3-x|")
    }

    func testScope() throws {
        let template = try Template("{% for item in seq %}{% endfor %}{{ item }}")
        let result = try template.render(["seq": Array(0 ... 9)])
        XCTAssertEqual(result, "")
    }

    func testVarlen() throws {
        let template = try Template("{% for item in iter %}{{ item }}{% endfor %}")
        let result = try template.render(["iter": Array(0 ... 4)])
        XCTAssertEqual(result, "01234")
    }

    func testNoniter() throws {
        let template = try Template("{% for item in none %}...{% endfor %}")
        XCTAssertThrowsError(try template.render(["none": nil]))
    }

    // TODO: Make this test pass
    //    func testRecursive() throws {
    //        let template = try Template(
    //            """
    //            {% for item in seq recursive -%}
    //            [{{ item.a }}{% if item.b %}<{{ loop(item.b) }}>{% endif %}]
    //            {%- endfor %}
    //            """
    //        )
    //
    //        let data: [String: Any] = [
    //            "seq": [
    //                ["a": 1, "b": [["a": 1], ["a": 2]]],
    //                ["a": 2, "b": [["a": 1], ["a": 2]]],
    //                ["a": 3, "b": [["a": "a"]]],
    //            ]
    //        ]
    //
    //        let result = try template.render(data)
    //        XCTAssertEqual(result, "[1<[1][2]>][2<[1][2]>][3<[a]>]")
    //    }

    func testLooploop() throws {
        let template = try Template(
            """
            {% for row in table %}
            {%- set rowloop = loop -%}
            {% for cell in row -%}
                [{{ rowloop.index }}|{{ loop.index }}]
            {%- endfor %}
            {%- endfor %}
            """
        )

        let result = try template.render(["table": ["ab", "cd"]])
        XCTAssertEqual(result, "[1|1][1|2][2|1][2|2]")
    }

    func testLoopFilter() throws {
        let template = try Template(
            "{% for item in range(10) if item is even %}[{{ item }}]{% endfor %}"
        )
        let result = try template.render([:])
        XCTAssertEqual(result, "[0][2][4][6][8]")

        let template2 = try Template(
            """
            {%- for item in range(10) if item is even %}[{{ loop.index }}:{{ item }}]{% endfor %}
            """
        )
        let result2 = try template2.render([:])
        XCTAssertEqual(result2, "[1:0][2:2][3:4][4:6][5:8]")
    }

    func testUnpacking() throws {
        let template = try Template(
            "{% for a, b, c in [[1, 2, 3]] %}{{ a }}|{{ b }}|{{ c }}{% endfor %}"
        )
        let result = try template.render([:])
        XCTAssertEqual(result, "1|2|3")
    }

    // TODO: Make this test pass
    //    func testRecursiveLookaround() throws {
    //        let template = try Template(
    //            """
    //            {% for item in seq recursive -%}
    //            [{{ loop.previtem.a if loop.previtem is defined else 'x' }}.\
    //            {{ item.a }}.\
    //            {{ loop.nextitem.a if loop.nextitem is defined else 'x' }}\
    //            {% if item.b %}<{{ loop(item.b) }}>{% endif %}]
    //            {%- endfor %}
    //            """
    //        )
    //
    //        let data: [String: Any] = [
    //            "seq": [
    //                ["a": 1, "b": [["a": 1], ["a": 2]]],
    //                ["a": 2, "b": [["a": 1], ["a": 2]]],
    //                ["a": 3, "b": [["a": "a"]]],
    //            ]
    //        ]
    //
    //        let result = try template.render(data)
    //        XCTAssertEqual(result, "[x.1.2<[x.1.2][1.2.x]>][1.2.3<[x.1.2][1.2.x]>][2.3.x<[x.a.x]>]")
    //    }

    // TODO: Make this test pass
    //    func testRecursiveDepth0() throws {
    //        let template = try Template(
    //            """
    //            {% for item in seq recursive -%}
    //            [{{ loop.depth0 }}:{{ item.a }}\
    //            {% if item.b %}<{{ loop(item.b) }}>{% endif %}]
    //            {%- endfor %}
    //            """
    //        )
    //
    //        let data: [String: Any] = [
    //            "seq": [
    //                ["a": 1, "b": [["a": 1], ["a": 2]]],
    //                ["a": 2, "b": [["a": 1], ["a": 2]]],
    //                ["a": 3, "b": [["a": "a"]]],
    //            ]
    //        ]
    //
    //        let result = try template.render(data)
    //        XCTAssertEqual(result, "[0:1<[1:1][1:2]>][0:2<[1:1][1:2]>][0:3<[1:a]>]")
    //    }

    // TODO: Make this test pass
    //    func testRecursiveDepth() throws {
    //        let template = try Template(
    //            """
    //            {% for item in seq recursive -%}
    //            [{{ loop.depth }}:{{ item.a }}\
    //            {% if item.b %}<{{ loop(item.b) }}>{% endif %}]
    //            {%- endfor %}
    //            """
    //        )
    //
    //        let data: [String: Any] = [
    //            "seq": [
    //                ["a": 1, "b": [["a": 1], ["a": 2]]],
    //                ["a": 2, "b": [["a": 1], ["a": 2]]],
    //                ["a": 3, "b": [["a": "a"]]],
    //            ]
    //        ]
    //
    //        let result = try template.render(data)
    //        XCTAssertEqual(result, "[1:1<[2:1][2:2]>][1:2<[2:1][2:2]>][1:3<[2:a]>]")
    //    }

    // TODO: Make this test pass
    //    func testReversedBug() throws {
    //        let template = try Template(
    //            """
    //            {% for i in items %}{{ i }}\
    //            {% if not loop.last %},{% endif %}\
    //            {% endfor %}
    //            """
    //        )
    //        let result = try template.render(["items": [3, 2, 1].reversed()])
    //        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "1,2,3")
    //    }

    // TODO: Make this test pass
    //    func testLoopErrors() throws {
    //        // Test accessing loop variable before loop starts
    //        let template1 = try Template(
    //            """
    //            {% for item in [1] if loop.index == 0 %}...{% endfor %}
    //            """
    //        )
    //        XCTAssertThrowsError(try template1.render([:]))
    //
    //        // Test accessing loop in else block
    //        let template2 = try Template(
    //            """
    //            {% for item in [] %}...{% else %}{{ loop }}{% endfor %}
    //            """
    //        )
    //        let result = try template2.render([:])
    //        XCTAssertEqual(result, "")
    //    }

    func testScopedSpecialVar() throws {
        let template = try Template(
            """
            {% for s in seq %}[{{ loop.first }}\
            {% for c in s %}|{{ loop.first }}{% endfor %}]\
            {% endfor %}
            """
        )
        let result = try template.render(["seq": ["ab", "cd"]])
        XCTAssertEqual(result, "[true|true|false][false|true|false]")
    }

    func testScopedLoopVar() throws {
        let template1 = try Template(
            """
            {% for x in seq %}{{ loop.first }}\
            {% for y in seq %}{% endfor %}\
            {% endfor %}
            """
        )
        let result1 = try template1.render(["seq": "ab"])
        XCTAssertEqual(result1, "truefalse")

        let template2 = try Template(
            """
            {% for x in seq %}\
            {% for y in seq %}{{ loop.first }}\
            {% endfor %}\
            {% endfor %}
            """
        )
        let result2 = try template2.render(["seq": "ab"])
        XCTAssertEqual(result2, "truefalsetruefalse")
    }

    // TODO: Make this test pass
    //    func testRecursiveEmptyLoopIter() throws {
    //        let template = try Template(
    //            """
    //            {%- for item in foo recursive -%}\
    //            {%- endfor -%}
    //            """
    //        )
    //        let result = try template.render(["foo": []])
    //        XCTAssertEqual(result, "")
    //    }

    // TODO: Make this test pass
    //    func testCallInLoop() throws {
    //        let template = try Template(
    //            """
    //            {%- macro do_something() -%}
    //                [{{ caller() }}]
    //            {%- endmacro %}
    //
    //            {%- for i in [1, 2, 3] %}
    //                {%- call do_something() -%}
    //                    {{ i }}
    //                {%- endcall %}
    //            {%- endfor -%}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "[1][2][3]")
    //    }
}

final class MacroTests: XCTestCase {
    func testSimpleMacro() throws {
        let template = try Template(
            """
            {% macro say_hello(name) %}Hello {{ name }}!{% endmacro %}
            {{ say_hello('Peter') }}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "Hello Peter!")
    }

    func testMacroScoping() throws {
        let template = try Template(
            """
            {% macro level1(data1) %}
            {% macro level2(data2) %}{{ data1 }}|{{ data2 }}{% endmacro %}
            {{ level2('bar') }}{% endmacro %}
            {{ level1('foo') }}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "foo|bar")
    }

    // TODO: Make this test pass
    //    func testMacroArguments() throws {
    //        let template = try Template(
    //            """
    //            {% macro m(a, b, c='c', d='d') %}{{ a }}|{{ b }}|{{ c }}|{{ d }}{% endmacro %}
    //            {{ m() }}|{{ m('a') }}|{{ m('a', 'b') }}|{{ m(1, 2, 3) }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "||c|d|a||c|d|a|b|c|d|1|2|3|d")
    //    }

    func testCallself() throws {
        let template = try Template(
            """
            {% macro foo(x) %}{{ x }}{% if x > 1 %}|{{ foo(x - 1) }}{% endif %}{% endmacro %}
            {{ foo(5) }}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "5|4|3|2|1")
    }

    // TODO: Make this test pass
    //    func testArgumentsDefaultsNonsense() throws {
    //        // Test that macro with invalid argument defaults throws error
    //        let template = try Template(
    //            """
    //            {% macro m(a, b=1, c) %}a={{ a }}, b={{ b }}, c={{ c }}{% endmacro %}
    //            """
    //        )
    //        XCTAssertThrowsError(try template.render([:]))
    //    }

    // TODO: Make this test pass
    //    func testCallerDefaultsNonsense() throws {
    //        let template = try Template(
    //            """
    //            {% macro a() %}{{ caller() }}{% endmacro %}
    //            {% call(x, y=1, z) a() %}{% endcall %}
    //            """
    //        )
    //        XCTAssertThrowsError(try template.render([:]))
    //    }

    // TODO: Make this test pass
    //    func testVarargs() throws {
    //        let template = try Template(
    //            """
    //            {% macro test() %}{{ varargs|join('|') }}{% endmacro %}\
    //            {{ test(1, 2, 3) }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "1|2|3")
    //    }

    // TODO: Make this test pass
    //    func testSimpleCall() throws {
    //        let template = try Template(
    //            """
    //            {% macro test() %}[[{{ caller() }}]]{% endmacro %}\
    //            {% call test() %}data{% endcall %}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "[[data]]")
    //    }

    // TODO: Make this test pass
    //    func testComplexCall() throws {
    //        let template = try Template(
    //            """
    //            {% macro test() %}[[{{ caller('data') }}]]{% endmacro %}\
    //            {% call(data) test() %}{{ data }}{% endcall %}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "[[data]]")
    //    }

    // TODO: Make this test pass
    //    func testCallerUndefined() throws {
    //        let template = try Template(
    //            """
    //            {% set caller = 42 %}\
    //            {% macro test() %}{{ caller is not defined }}{% endmacro %}\
    //            {{ test() }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "true")
    //    }
}

final class SetTests: XCTestCase {
    // MARK: - Set Tests

    func testNormalSet() throws {
        let template = try Template("{% set foo = 1 %}{{ foo }}")
        let result = try template.render([:])
        XCTAssertEqual(result, "1")
    }

    // TODO: Make this test pass
    //    func testBlockSet() throws {
    //        let template = try Template("{% set foo %}42{% endset %}{{ foo }}")
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "42")
    //    }

    func testNamespace() throws {
        let template = try Template(
            """
            {% set ns = namespace() %}{% set ns.bar = '42' %}{{ ns.bar }}
            """
        )
        let result = try template.render([:])
        XCTAssertEqual(result, "42")
    }

    // TODO: Make this test pass
    //    func testNamespaceLoop() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace(found=false) %}\
    //            {% for x in range(4) %}\
    //            {% if x == v %}\
    //            {% set ns.found = true %}\
    //            {% endif %}\
    //            {% endfor %}\
    //            {{ ns.found }}
    //            """
    //        )
    //
    //        let result1 = try template.render(["v": 3])
    //        XCTAssertEqual(result1, "true")
    //
    //        let result2 = try template.render(["v": 4])
    //        XCTAssertEqual(result2, "false")
    //    }

    // TODO: Make this test pass
    //    func testNamespaceBlock() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace() %}{% set ns.bar %}42{% endset %}{{ ns.bar }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result, "42")
    //    }

    // TODO: Make this test pass
    //    func testInitNamespace() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace(d, self=37) %}
    //            {% set ns.b = 42 %}
    //            {{ ns.a }}|{{ ns.self }}|{{ ns.b }}
    //            """
    //        )
    //        let result = try template.render(["d": ["a": 13]])
    //        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "13|37|42")
    //    }

    // TODO: Make this test pass
    //    func testNamespaceMacro() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace() %}
    //            {% set ns.a = 13 %}
    //            {% macro magic(x) %}
    //            {% set x.b = 37 %}
    //            {% endmacro %}
    //            {{ magic(ns) }}
    //            {{ ns.a }}|{{ ns.b }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "13|37")
    //    }

    // TODO: Make this test pass
    //    func testNamespaceSetTuple() throws {
    //        let template = try Template(
    //            """
    //            {% set ns = namespace(a=12, b=36) %}
    //            {% set ns.a, ns.b = ns.a + 1, ns.b + 1 %}
    //            {{ ns.a }}|{{ ns.b }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "13|37")
    //    }

    // TODO: Make this test pass
    //    func testBlockEscaping() throws {
    //        let template = try Template(
    //            """
    //            {% set foo %}<em>{{ test }}</em>{% endset %}\
    //            foo: {{ foo }}
    //            """
    //        )
    //        let result = try template.render(["test": "<unsafe>"])
    //        XCTAssertEqual(
    //            result.trimmingCharacters(in: .whitespaces),
    //            "foo: <em>&lt;unsafe&gt;</em>"
    //        )
    //    }

    // TODO: Make this test pass
    //    func testBlockEscapingFiltered() throws {
    //        let template = try Template(
    //            """
    //            {% set foo | trim %}<em>{{ test }}</em>    {% endset %}\
    //            foo: {{ foo }}
    //            """
    //        )
    //        let result = try template.render(["test": "<unsafe>"])
    //        XCTAssertEqual(
    //            result.trimmingCharacters(in: .whitespaces),
    //            "foo: <em>&lt;unsafe&gt;</em>"
    //        )
    //    }

    // TODO: Make this test pass
    //    func testBlockFiltered() throws {
    //        let template = try Template(
    //            """
    //            {% set foo | trim | length | string %} 42    {% endset %}\
    //            {{ foo }}
    //            """
    //        )
    //        let result = try template.render([:])
    //        XCTAssertEqual(result.trimmingCharacters(in: .whitespaces), "2")
    //    }

    // TODO: Make this test pass
    //    func testSetInvalid() throws {
    //        // Test invalid set syntax
    //        let template1 = try Template("{% set foo['bar'] = 1 %}")
    //        XCTAssertThrowsError(try template1.render([:]))
    //
    //        // Test setting attribute on non-namespace
    //        let template2 = try Template("{% set foo.bar = 1 %}")
    //        XCTAssertThrowsError(try template2.render(["foo": [:]]))
    //    }

    func testNamespaceRedefined() throws {
        let template = try Template(
            """
            {% set ns = namespace() %}\
            {% set ns.bar = 'hi' %}
            """
        )
        XCTAssertThrowsError(try template.render(["namespace": [String: Any].self]))
    }
}

// TODO: Make these tests pass
//final class WithTests: XCTestCase {
//    func testWith() throws {
//        let template = try Template(
//            """
//            {% with a=42, b=23 -%}
//                {{ a }} = {{ b }}
//            {% endwith -%}
//                {{ a }} = {{ b }}
//            """
//        )
//        let result = try template.render(["a": 1, "b": 2])
//        let lines = result.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
//        XCTAssertEqual(lines, ["42 = 23", "1 = 2"])
//    }
//
//    func testWithArgumentScoping() throws {
//        let template = try Template(
//            """
//            {%- with a=1, b=2, c=b, d=e, e=5 -%}
//                {{ a }}|{{ b }}|{{ c }}|{{ d }}|{{ e }}
//            {%- endwith -%}
//            """
//        )
//        let result = try template.render(["b": 3, "e": 4])
//        XCTAssertEqual(result, "1|2|3|4|5")
//    }
//}
