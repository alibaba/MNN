/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import XCTest
import Splash

/// Test case used as an abstract base class for all tests relating to
/// syntax highlighting. For all such tests, the Swift grammar is used.
class SyntaxHighlighterTestCase: XCTestCase {
    private(set) var highlighter: SyntaxHighlighter<OutputFormatMock>!
    private(set) var builder: OutputBuilderMock!

    override func setUp() {
        super.setUp()
        builder = OutputBuilderMock()
        highlighter = SyntaxHighlighter(format: OutputFormatMock(builder: builder))
    }
}
