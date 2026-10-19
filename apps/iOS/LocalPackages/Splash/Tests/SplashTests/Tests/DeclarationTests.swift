/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

final class DeclarationTests: SyntaxHighlighterTestCase {
    func testFunctionDeclaration() {
        let components = highlighter.highlight("func hello(world: String) -> Int")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello(world:"),
            .whitespace(" "),
            .token("String", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Int", .type)
        ])
    }

    func testRequiredFunctionDeclaration() {
        let components = highlighter.highlight("required func hello()")

        XCTAssertEqual(components, [
            .token("required", .keyword),
            .whitespace(" "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello()")
        ])
    }

    func testPublicFunctionDeclarationWithDocumentationEndingWithDot() {
        let components = highlighter.highlight("""
        /// Documentation.
        public func hello()
        """)

        XCTAssertEqual(components, [
            .token("///", .comment),
            .whitespace(" "),
            .token("Documentation.", .comment),
            .whitespace("\n"),
            .token("public", .keyword),
            .whitespace(" "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello()")
        ])
    }

    func testFunctionDeclarationWithEmptyExternalLabel() {
        let components = highlighter.highlight("func a(_ b: B)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("a("),
            .token("_", .keyword),
            .whitespace(" "),
            .plainText("b:"),
            .whitespace(" "),
            .token("B", .type),
            .plainText(")")
        ])
    }

    func testFunctionDeclarationWithKeywordArgumentLabel() {
        let components = highlighter.highlight("func a(for b: B)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("a(for"),
            .whitespace(" "),
            .plainText("b:"),
            .whitespace(" "),
            .token("B", .type),
            .plainText(")")
        ])
    }

    func testFunctionDeclarationWithKeywordArgumentLabelOnNewLine() {
        let components = highlighter.highlight("""
        func a(
            for b: B
        )
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("a("),
            .whitespace("\n    "),
            .plainText("for"),
            .whitespace(" "),
            .plainText("b:"),
            .whitespace(" "),
            .token("B", .type),
            .whitespace("\n"),
            .plainText(")")
        ])
    }

    func testGenericFunctionDeclarationWithKeywordArgumentLabel() {
        let components = highlighter.highlight("func perform<O: AnyObject>(for object: O) {}")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("perform<O:"),
            .whitespace(" "),
            .token("AnyObject", .type),
            .plainText(">(for"),
            .whitespace(" "),
            .plainText("object:"),
            .whitespace(" "),
            .token("O", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testGenericFunctionDeclarationWithoutConstraints() {
        let components = highlighter.highlight("func hello<A, B>(a: A, b: B)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello<A,"),
            .whitespace(" "),
            .plainText("B>(a:"),
            .whitespace(" "),
            .token("A", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("b:"),
            .whitespace(" "),
            .token("B", .type),
            .plainText(")")
        ])
    }

    func testGenericFunctionDeclarationWithSingleConstraint() {
        let components = highlighter.highlight("func hello<T: AnyObject>(t: T)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello<T:"),
            .whitespace(" "),
            .token("AnyObject", .type),
            .plainText(">(t:"),
            .whitespace(" "),
            .token("T", .type),
            .plainText(")")
        ])
    }

    func testGenericFunctionDeclarationWithMultipleConstraints() {
        let components = highlighter.highlight("func hello<A: AnyObject, B: Sequence>(a: A, b: B)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello<A:"),
            .whitespace(" "),
            .token("AnyObject", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("B:"),
            .whitespace(" "),
            .token("Sequence", .type),
            .plainText(">(a:"),
            .whitespace(" "),
            .token("A", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("b:"),
            .whitespace(" "),
            .token("B", .type),
            .plainText(")")
        ])
    }

    func testGenericFunctionDeclarationWithGenericParameter() {
        let components = highlighter.highlight("func value<T>(at keyPath: KeyPath<Element, T>) -> T? {}")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("value<T>(at"),
            .whitespace(" "),
            .plainText("keyPath:"),
            .whitespace(" "),
            .token("KeyPath", .type),
            .plainText("<"),
            .token("Element", .type),
            .plainText(","),
            .whitespace(" "),
            .token("T", .type),
            .plainText(">)"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("T", .type),
            .plainText("?"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testFunctionDeclarationWithGenericReturnType() {
        let components = highlighter.highlight("""
        func array() -> Array<Element> { return [] }
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("array()"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Array", .type),
            .plainText("<"),
            .token("Element", .type),
            .plainText(">"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("return", .keyword),
            .whitespace(" "),
            .plainText("[]"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testGenericStructDeclaration() {
        let components = highlighter.highlight("struct MyStruct<A: Hello, B> {}")

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("MyStruct<A:"),
            .whitespace(" "),
            .token("Hello", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("B>"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testClassDeclaration() {
        let components = highlighter.highlight("""
        class Hello {
            var required: String
            var optional: Int?
        }
        """)

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("required:"),
            .whitespace(" "),
            .token("String", .type),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("optional:"),
            .whitespace(" "),
            .token("Int", .type),
            .plainText("?"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testCompactClassDeclarationWithInitializer() {
        let components = highlighter.highlight("class Foo { init(hello: Int) {} }")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Foo"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("init", .keyword),
            .plainText("(hello:"),
            .whitespace(" "),
            .token("Int", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testClassDeclarationWithDeinit() {
        let components = highlighter.highlight("class Foo { deinit {} }")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Foo"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("deinit", .keyword),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testClassDeclarationWithMultipleProtocolConformances() {
        let components = highlighter.highlight("class MyClass: ProtocolA, ProtocolB {}")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("MyClass:"),
            .whitespace(" "),
            .token("ProtocolA", .type),
            .plainText(","),
            .whitespace(" "),
            .token("ProtocolB", .type),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testSubclassDeclaration() {
        let components = highlighter.highlight("class ViewController: UIViewController { }")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("ViewController:"),
            .whitespace(" "),
            .token("UIViewController", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testGenericSubclassDeclaration() {
        let components = highlighter.highlight("class Promise<Value>: Future<Value> {}")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Promise<Value>:"),
            .whitespace(" "),
            .token("Future", .type),
            .plainText("<"),
            .token("Value", .type),
            .plainText(">"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testProtocolDeclaration() {
        let components = highlighter.highlight("""
        protocol Hello {
            var property: String { get set }
            func method()
        }
        """)

        XCTAssertEqual(components, [
            .token("protocol", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property:"),
            .whitespace(" "),
            .token("String", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("get", .keyword),
            .whitespace(" "),
            .token("set", .keyword),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n    "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("method()"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testProtocolDeclarationWithAssociatedTypes() {
        let components = highlighter.highlight("""
        protocol Task {
            associatedtype Input
            associatedtype Error: Swift.Error
        }
        """)

        XCTAssertEqual(components, [
            .token("protocol", .keyword),
            .whitespace(" "),
            .plainText("Task"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("associatedtype", .keyword),
            .whitespace(" "),
            .plainText("Input"),
            .whitespace("\n    "),
            .token("associatedtype", .keyword),
            .whitespace(" "),
            .plainText("Error:"),
            .whitespace(" "),
            .token("Swift", .type),
            .plainText("."),
            .token("Error", .type),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testExtensionDeclaration() {
        let components = highlighter.highlight("extension UIViewController { }")

        XCTAssertEqual(components, [
            .token("extension", .keyword),
            .whitespace(" "),
            .token("UIViewController", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testExtensionDeclarationWithConvenienceInitializer() {
        let components = highlighter.highlight("""
        extension Node { convenience init(name: String) { self.init() } }
        """)

        XCTAssertEqual(components, [
            .token("extension", .keyword),
            .whitespace(" "),
            .token("Node", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("convenience", .keyword),
            .whitespace(" "),
            .token("init", .keyword),
            .plainText("(name:"),
            .whitespace(" "),
            .token("String", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("self", .keyword),
            .plainText("."),
            .token("init", .keyword),
            .plainText("()"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testExtensionDeclarationWithConstraint() {
        let components = highlighter.highlight("extension Hello where Foo == String, Bar: Numeric { }")

        XCTAssertEqual(components, [
            .token("extension", .keyword),
            .whitespace(" "),
            .token("Hello", .type),
            .whitespace(" "),
            .token("where", .keyword),
            .whitespace(" "),
            .token("Foo", .type),
            .whitespace(" "),
            .plainText("=="),
            .whitespace(" "),
            .token("String", .type),
            .plainText(","),
            .whitespace(" "),
            .token("Bar", .type),
            .plainText(":"),
            .whitespace(" "),
            .token("Numeric", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testLazyPropertyDeclaration() {
        let components = highlighter.highlight("""
        struct Hello {
            lazy var property = 0
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("lazy", .keyword),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("0", .number),
            .whitespace("\n"),
            .plainText("}")
        ])
    }
    
    func testDynamicPropertyDeclaration() {
        let components = highlighter.highlight("""
        class Hello {
            @objc dynamic var property = 0
        }
        """)
        
        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("@objc", .keyword),
            .whitespace(" "),
            .token("dynamic", .keyword),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("0", .number),
            .whitespace("\n"),
            .plainText("}")
        ])
    }


    func testGenericPropertyDeclaration() {
        let components = highlighter.highlight("class Hello { var array: Array<String> = [] }")

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("array:"),
            .whitespace(" "),
            .token("Array", .type),
            .plainText("<"),
            .token("String", .type),
            .plainText(">"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("[]"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testPropertyDeclarationWithWillSet() {
        let components = highlighter.highlight("""
        struct Hello {
            var property: Int { willSet { } }
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property:"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("willSet", .keyword),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPropertyDeclarationWithDidSet() {
        let components = highlighter.highlight("""
        struct Hello {
            var property: Int { didSet { } }
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property:"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("didSet", .keyword),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPropertyWithCommentedDidSet() {
        let components = highlighter.highlight("""
        struct Hello {
            var property: Int {
                // Comment.
                didSet { }
            }
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property:"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n        "),
            .token("//", .comment),
            .whitespace(" "),
            .token("Comment.", .comment),
            .whitespace("\n        "),
            .token("didSet", .keyword),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n    "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPropertyWithSetterAccessLevel() {
        let components = highlighter.highlight("""
        struct Hello {
            private(set) var property: Int
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Hello"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("private(set)", .keyword),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("property:"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPropertyDeclarationAfterCommentEndingWithVarKeyword() {
        let components = highlighter.highlight("""
        // var
        var number = 7
        """)

        XCTAssertEqual(components, [
            .token("//", .comment),
            .whitespace(" "),
            .token("var", .comment),
            .whitespace("\n"),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("number"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("7", .number)
        ])
    }

    func testPropertyDeclarationWithStaticPropertyDefaultValue() {
        let components = highlighter.highlight("""
        class ViewModel {
            var state = LoadingState<Output>.idle
        }
        """)

        XCTAssertEqual(components, [
            .token("class", .keyword),
            .whitespace(" "),
            .plainText("ViewModel"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("state"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("LoadingState", .type),
            .plainText("<"),
            .token("Output", .type),
            .plainText(">."),
            .token("idle", .property),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testSubscriptDeclaration() {
        let components = highlighter.highlight("""
        extension Collection {
            subscript(key: Key) -> Value? { return nil }
        }
        """)

        XCTAssertEqual(components, [
            .token("extension", .keyword),
            .whitespace(" "),
            .token("Collection", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("subscript", .keyword),
            .plainText("(key:"),
            .whitespace(" "),
            .token("Key", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Value", .type),
            .plainText("?"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("return", .keyword),
            .whitespace(" "),
            .token("nil", .keyword),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testGenericSubscriptDeclaration() {
        let components = highlighter.highlight("""
        extension Collection {
            subscript<T>(key: Key<T>) -> T? { return nil }
        }
        """)

        XCTAssertEqual(components, [
            .token("extension", .keyword),
            .whitespace(" "),
            .token("Collection", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("subscript", .keyword),
            .plainText("<T>(key:"),
            .whitespace(" "),
            .token("Key", .type),
            .plainText("<"),
            .token("T", .type),
            .plainText(">)"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("T", .type),
            .plainText("?"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("return", .keyword),
            .whitespace(" "),
            .token("nil", .keyword),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testDeferDeclaration() {
        let components = highlighter.highlight("func hello() { defer {} }")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("hello()"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("defer", .keyword),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testFunctionDeclarationWithInOutParameter() {
        let components = highlighter.highlight("func swapValues(value1: inout Int, value2: inout Int) { }")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("swapValues(value1:"),
            .whitespace(" "),
            .token("inout", .keyword),
            .whitespace(" "),
            .token("Int", .type),
            .plainText(","),
            .whitespace(" "),
            .plainText("value2:"),
            .whitespace(" "),
            .token("inout", .keyword),
            .whitespace(" "),
            .token("Int", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testFunctionDeclarationWithIgnoredParameter() {
        let components = highlighter.highlight("func perform(with _: Void) {}")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("perform(with"),
            .whitespace(" "),
            .token("_", .keyword),
            .plainText(":"),
            .whitespace(" "),
            .token("Void", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testFunctionDeclarationWithNonEscapedKeywordAsName() {
        let components = highlighter.highlight("func get() -> Int { return 7 }")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("get()"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("return", .keyword),
            .whitespace(" "),
            .token("7", .number),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testFunctionDeclarationWithEscapedKeywordAsName() {
        let components = highlighter.highlight("func `public`() -> Int { return 7 }")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("`public`()"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("return", .keyword),
            .whitespace(" "),
            .token("7", .number),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testFunctionDeclarationWithPreProcessors() {
        let components = highlighter.highlight("""
        func log(_ file: StaticString = #file, _ function: StaticString = #function) {}
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("log("),
            .token("_", .keyword),
            .whitespace(" "),
            .plainText("file:"),
            .whitespace(" "),
            .token("StaticString", .type),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("#file", .keyword),
            .plainText(","),
            .whitespace(" "),
            .token("_", .keyword),
            .whitespace(" "),
            .plainText("function:"),
            .whitespace(" "),
            .token("StaticString", .type),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("#function", .keyword),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testNonMutatingFunction() {
        let components = highlighter.highlight("""
        struct MyStruct {
            nonmutating func doNotChangeState() { }
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("MyStruct"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("nonmutating", .keyword),
            .whitespace(" "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("doNotChangeState()"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testRethrowingFunctionDeclaration() {
        let components = highlighter.highlight("""
        func map<T>(_ transform: (Element) throws -> T) rethrows -> [T]
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("map<T>("),
            .token("_", .keyword),
            .whitespace(" "),
            .plainText("transform:"),
            .whitespace(" "),
            .plainText("("),
            .token("Element", .type),
            .plainText(")"),
            .whitespace(" "),
            .token("throws", .keyword),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("T", .type),
            .plainText(")"),
            .whitespace(" "),
            .token("rethrows", .keyword),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .plainText("["),
            .token("T", .type),
            .plainText("]")
        ])
    }

    func testFunctionDeclarationWithOpaqueReturnType() {
        let components = highlighter.highlight(#"func make() -> some View { Text("!") }"#)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("make()"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("some", .keyword),
            .whitespace(" "),
            .token("View", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("Text", .type),
            .plainText("("),
            .token(#""!""#, .string),
            .plainText(")"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testFunctionDeclarationWithAnyParameter() {
        let components = highlighter.highlight("func process(value: any Value)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("process(value:"),
            .whitespace(" "),
            .token("any", .keyword),
            .whitespace(" "),
            .token("Value", .type),
            .plainText(")")
        ])
    }

    func testFunctionDeclarationWithGenericAnyParameter() {
        let components = highlighter.highlight("func process(collection: any Collection<any Value>)")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("process(collection:"),
            .whitespace(" "),
            .token("any", .keyword),
            .whitespace(" "),
            .token("Collection", .type),
            .plainText("<"),
            .token("any", .keyword),
            .whitespace(" "),
            .token("Value", .type),
            .plainText(">)")
        ])
    }

    func testFunctionDeclarationWithAnyDictionary() {
        let components = highlighter.highlight("func process(dictionary: [String: any Value])")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("process(dictionary:"),
            .whitespace(" "),
            .plainText("["),
            .token("String", .type),
            .plainText(":"),
            .whitespace(" "),
            .token("any", .keyword),
            .whitespace(" "),
            .token("Value", .type),
            .plainText("])")
        ])
    }

    func testPrefixFunctionDeclaration() {
        let components = highlighter.highlight("prefix func !(rhs: Bool) -> Bool { !rhs }")

        XCTAssertEqual(components, [
            .token("prefix", .keyword),
            .whitespace(" "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("!(rhs:"),
            .whitespace(" "),
            .token("Bool", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Bool", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .plainText("!rhs"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testEnumDeclarationWithSomeCase() {
        let components = highlighter.highlight("""
        enum MyEnum { case some }
        """)

        XCTAssertEqual(components, [
            .token("enum", .keyword),
            .whitespace(" "),
            .plainText("MyEnum"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("some"),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testIndirectEnumDeclaration() {
        let components = highlighter.highlight("""
        indirect enum Content {
            case single(String)
            case collection([Content])
        }
        """)

        XCTAssertEqual(components, [
            .token("indirect", .keyword),
            .whitespace(" "),
            .token("enum", .keyword),
            .whitespace(" "),
            .plainText("Content"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("single("),
            .token("String", .type),
            .plainText(")"),
            .whitespace("\n    "),
            .token("case", .keyword),
            .whitespace(" "),
            .plainText("collection(["),
            .token("Content", .type),
            .plainText("])"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPropertyWrapperDeclaration() {
        let components = highlighter.highlight("""
        @propertyWrapper
        struct Wrapped<Value> {
            var wrappedValue: Value
        }
        """)

        XCTAssertEqual(components, [
            .token("@propertyWrapper", .keyword),
            .whitespace("\n"),
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Wrapped<Value>"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("wrappedValue:"),
            .whitespace(" "),
            .token("Value", .type),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testWrappedPropertyDeclarations() {
        let components = highlighter.highlight("""
        struct User {
            @Persisted(key: "name") var name: String
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("User"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("@Persisted", .keyword),
            .plainText("(key:"),
            .whitespace(" "),
            .token(#""name""#, .string),
            .plainText(")"),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("name:"),
            .whitespace(" "),
            .token("String", .type),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testWrappedPropertyDeclarationUsingNestedType() {
        let components = highlighter.highlight("""
        struct User {
            @Persisted.InMemory var name: String
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("User"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("@Persisted", .keyword),
            .plainText("."),
            .token("InMemory", .keyword),
            .whitespace(" "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("name:"),
            .whitespace(" "),
            .token("String", .type),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testWrappedPropertyDeclarationUsingExplicitType() {
        let components = highlighter.highlight("""
        struct Model {
            @Wrapper<Bool>(key: "setting")
            var setting
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Model"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("@Wrapper", .keyword),
            .plainText("<"),
            .token("Bool", .type),
            .plainText(">(key:"),
            .whitespace(" "),
            .token(#""setting""#, .string),
            .plainText(")"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("setting"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testGenericInitializerDeclaration() {
        let components = highlighter.highlight("""
        struct Box {
            init<T: Model>(model: T) {}
        }
        """)

        XCTAssertEqual(components, [
            .token("struct", .keyword),
            .whitespace(" "),
            .plainText("Box"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("init", .keyword),
            .plainText("<T:"),
            .whitespace(" "),
            .token("Model", .type),
            .plainText(">(model:"),
            .whitespace(" "),
            .token("T", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testNonThrowingAsyncFunctionDeclaration() {
        let components = highlighter.highlight("func test() async {}")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("test()"),
            .whitespace(" "),
            .token("async", .keyword),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testNonThrowingAsyncFunctionDeclarationWithReturnValue() {
        let components = highlighter.highlight("func test() async -> Int { 0 }")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("test()"),
            .whitespace(" "),
            .token("async", .keyword),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("Int", .type),
            .whitespace(" "),
            .plainText("{"),
            .whitespace(" "),
            .token("0", .number),
            .whitespace(" "),
            .plainText("}")
        ])
    }

    func testThrowingAsyncFunctionDeclaration() {
        let components = highlighter.highlight("func test() async throws {}")

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("test()"),
            .whitespace(" "),
            .token("async", .keyword),
            .whitespace(" "),
            .token("throws", .keyword),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testDeclaringGenericFunctionNamedAwait() {
        let components = highlighter.highlight("""
        func await<T>(_ function: () -> T) {}
        """)

        XCTAssertEqual(components, [
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("await<T>("),
            .token("_", .keyword),
            .whitespace(" "),
            .plainText("function:"),
            .whitespace(" "),
            .plainText("()"),
            .whitespace(" "),
            .plainText("->"),
            .whitespace(" "),
            .token("T", .type),
            .plainText(")"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testActorDeclaration() {
        let components = highlighter.highlight("""
        actor MyActor {
            var value = 0
            func action() {}
        }
        """)

        XCTAssertEqual(components, [
            .token("actor", .keyword),
            .whitespace(" "),
            .plainText("MyActor"),
            .whitespace(" "),
            .plainText("{"),
            .whitespace("\n    "),
            .token("var", .keyword),
            .whitespace(" "),
            .plainText("value"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("0", .number),
            .whitespace("\n    "),
            .token("func", .keyword),
            .whitespace(" "),
            .plainText("action()"),
            .whitespace(" "),
            .plainText("{}"),
            .whitespace("\n"),
            .plainText("}")
        ])
    }

    func testPublicActorDeclaration() {
        let components = highlighter.highlight("public actor MyActor {}")

        XCTAssertEqual(components, [
            .token("public", .keyword),
            .whitespace(" "),
            .token("actor", .keyword),
            .whitespace(" "),
            .plainText("MyActor"),
            .whitespace(" "),
            .plainText("{}")
        ])
    }

    func testDeclaringAndMutatingLocalVariableNamedActor() {
        let components = highlighter.highlight("""
        let actor = Actor()
        actor.position = scene.center
        """)

        XCTAssertEqual(components, [
            .token("let", .keyword),
            .whitespace(" "),
            .plainText("actor"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .token("Actor", .type),
            .plainText("()"),
            .whitespace("\n"),
            .plainText("actor."),
            .token("position", .property),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("scene."),
            .token("center", .property)
        ])
    }

    func testPassingAndReferencingLocalVariableNamedActor() {
        let components = highlighter.highlight("""
        prepare(actor: actor)
        scene.add(actor)
        latestActor = actor
        return actor
        """)

        XCTAssertEqual(components, [
            .token("prepare", .call),
            .plainText("(actor:"),
            .whitespace(" "),
            .plainText("actor)"),
            .whitespace("\n"),
            .plainText("scene."),
            .token("add", .call),
            .plainText("(actor)"),
            .whitespace("\n"),
            .plainText("latestActor"),
            .whitespace(" "),
            .plainText("="),
            .whitespace(" "),
            .plainText("actor"),
            .whitespace("\n"),
            .token("return", .keyword),
            .whitespace(" "),
            .plainText("actor")
        ])
    }
}
