import Foundation
import XCTest
import Splash

final class HTMLOutputFormatTests: XCTestCase {
    private var highlighter: SyntaxHighlighter<HTMLOutputFormat>!

    override func setUp() {
        super.setUp()
        highlighter = SyntaxHighlighter(format: HTMLOutputFormat())
    }

    func testBasicGeneration() {
        let html = highlighter.highlight("""
        public struct Test: SomeProtocol {
            func hello() -> Int { return 7 }
        }
        """)

        XCTAssertEqual(html, """
        <span class="keyword">public struct</span> Test: <span class="type">SomeProtocol</span> {
            <span class="keyword">func</span> hello() -&gt; <span class="type">Int</span> { <span class="keyword">return</span> <span class="number">7</span> }
        }
        """)
    }

    func testStrippingGreaterAndLessThanCharactersFromOutput() {
        let html = highlighter.highlight("Array<String>")

        XCTAssertEqual(html, """
        <span class="type">Array</span>&lt;<span class="type">String</span>&gt;
        """)
    }

    func testCommentMerging() {
        let html = highlighter.highlight("// Hey I'm a comment!")

        XCTAssertEqual(html, """
        <span class="comment">// Hey I'm a comment!</span>
        """)
    }
}
