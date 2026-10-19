/**
 *  Splash
 *  Copyright (c) John Sundell 2019
 *  MIT license - see LICENSE.md
 */

import XCTest
import Splash

final class MarkdownTests: XCTestCase {
    private var decorator: MarkdownDecorator!

    override func setUp() {
        super.setUp()
        decorator = MarkdownDecorator()
    }

    func testConvertingCodeBlock() {
        let markdown = """
        # Title

        Text text text `inline.code.shouldNotBeHighlighted()`.

        ```
        struct Hello: Protocol {}
        ```

        Text.
        """

        let expectedResult = """
        # Title

        Text text text `inline.code.shouldNotBeHighlighted()`.

        <pre class="splash"><code><span class="keyword">struct</span> Hello: <span class="type">Protocol</span> {}</code></pre>

        Text.
        """

        XCTAssertEqual(decorator.decorate(markdown), expectedResult)
    }

    func testSkippingHighlightingForCodeBlock() {
        let markdown = """
        Text text.

        ```no-highlight
        struct Hello: Protocol {}
        ```

        Text.
        """

            let expectedResult = """
            Text text.

            <pre class="splash"><code>struct Hello: Protocol {}</code></pre>

            Text.
            """

        XCTAssertEqual(decorator.decorate(markdown), expectedResult)
    }

    func testEscapingSpecialCharactersWithinHighlightedCodeBlock() {
        let markdown = """
        Text text.

        ```
        let a = "<Hello&World>"
        ```

        Text.
        """

        let expectedResult = """
        Text text.

        <pre class="splash"><code><span class="keyword">let</span> a = <span class="string">"&lt;Hello&amp;World&gt;"</span></code></pre>

        Text.
        """

        XCTAssertEqual(decorator.decorate(markdown), expectedResult)
    }

    func testEscapingSpecialCharactersWithinSkippedCodeBlock() {
        let markdown = """
        Text text.

        ```no-highlight
        let a = "<Hello&World>"
        ```

        Text.
        """

        let expectedResult = """
        Text text.

        <pre class="splash"><code>let a = "&lt;Hello&amp;World&gt;"</code></pre>

        Text.
        """

        XCTAssertEqual(decorator.decorate(markdown), expectedResult)
    }
}
