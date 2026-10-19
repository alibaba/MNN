/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import Splash

guard CommandLine.arguments.count > 1 else {
    print("⚠️  Please supply the code to tokenize as a string argument")
    exit(1)
}

let code = CommandLine.arguments[1]
let highlighter = SyntaxHighlighter(format: TokenizerOutputFormat())
print(highlighter.highlight(code))
