//
//  InterpreterTests.swift
//
//
//  Created by John Mai on 2024/3/23.
//

import XCTest

@testable import Jinja

let exampleIfTemplate = "<div>\n    {% if True %}\n        yay\n    {% endif %}\n</div>"

let exampleForTemplate = "{% for item in seq %}\n    {{ item }}\n{% endfor %}"
let exampleForTemplate2 = "{% for item in seq -%}\n    {{ item }}\n{% endfor %}"
let exampleForTemplate3 = "{% for item in seq %}\n    {{ item }}\n{%- endfor %}"
let exampleForTemplate4 = "{% for item in seq -%}\n    {{ item }}\n{%- endfor %}"

let exampleCommentTemplate = "    {# comment #}\n  {# {% if true %} {% endif %} #}\n"

let seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]

final class InterpreterTests: XCTestCase {
    struct Test {
        let template: String
        let data: [String: Any]
        var options: PreprocessOptions = .init()
        let target: String
    }

    let tests: [Test] = [
        Test(
            template: exampleIfTemplate,
            data: [:],
            target: "<div>\n    \n        yay\n    \n</div>"
        ),
        Test(
            template: exampleIfTemplate,
            data: [:],
            options: .init(lstripBlocks: true),
            target: "<div>\n\n        yay\n\n</div>"
        ),
        Test(
            template: exampleIfTemplate,
            data: [:],
            options: .init(trimBlocks: true),
            target: "<div>\n            yay\n    </div>"
        ),
        Test(
            template: exampleIfTemplate,
            data: [:],
            options: .init(trimBlocks: true, lstripBlocks: true),
            target: "<div>\n        yay\n</div>"
        ),
        Test(
            template: exampleForTemplate,
            data: [
                "seq": seq
            ],
            target: "\n    1\n\n    2\n\n    3\n\n    4\n\n    5\n\n    6\n\n    7\n\n    8\n\n    9\n"
        ),
        Test(
            template: exampleForTemplate,
            data: [
                "seq": seq
            ],
            options: .init(lstripBlocks: true),
            target: "\n    1\n\n    2\n\n    3\n\n    4\n\n    5\n\n    6\n\n    7\n\n    8\n\n    9\n"
        ),
        Test(
            template: exampleForTemplate,
            data: [
                "seq": seq
            ],
            options: .init(trimBlocks: true),
            target: "    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n"
        ),
        Test(
            template: exampleForTemplate,
            data: [
                "seq": seq
            ],
            options: .init(trimBlocks: true, lstripBlocks: true),
            target: "    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n"
        ),
        Test(
            template: exampleForTemplate2,
            data: [
                "seq": seq
            ],
            target: "1\n2\n3\n4\n5\n6\n7\n8\n9\n"
        ),
        Test(
            template: exampleForTemplate3,
            data: [
                "seq": seq
            ],
            target: "\n    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9"
        ),
        Test(
            template: exampleForTemplate3,
            data: [
                "seq": seq
            ],
            options: .init(trimBlocks: true),
            target: "    1    2    3    4    5    6    7    8    9"
        ),
        Test(
            template: exampleForTemplate4,
            data: [
                "seq": seq
            ],
            target: "123456789"
        ),
        Test(
            template: exampleCommentTemplate,
            data: [:],
            target: "    \n  "
        ),
        Test(
            template: exampleCommentTemplate,
            data: [:],
            options: .init(lstripBlocks: true),
            target: "\n"
        ),
        Test(
            template: exampleCommentTemplate,
            data: [:],
            options: .init(trimBlocks: true),
            target: "      "
        ),
        Test(
            template: exampleCommentTemplate,
            data: [:],
            options: .init(trimBlocks: true, lstripBlocks: true),
            target: ""
        ),
    ]

    func testRender() throws {
        for test in tests {
            let env = Environment()
            try env.set(name: "True", value: true)
            for (key, value) in test.data {
                try env.set(name: key, value: value)
            }
            let tokens = try tokenize(test.template, options: test.options)
            let parsed = try parse(tokens: tokens)
            let interpreter = Interpreter(env: env)
            let result = try interpreter.run(program: parsed)
            if let stringResult = result as? StringValue {
                XCTAssertEqual(stringResult.value.debugDescription, test.target.debugDescription)
            } else {
                XCTFail("Expected a StringValue, but got \(type(of: result))")
            }
        }
    }
}
