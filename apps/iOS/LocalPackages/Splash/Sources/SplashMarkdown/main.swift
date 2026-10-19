/**
 *  Splash
 *  Copyright (c) John Sundell 2019
 *  MIT license - see LICENSE.md
 */

import Foundation
import Splash

guard CommandLine.arguments.count > 1 else {
    print("⚠️  Please supply the path to a Markdown file to process as an argument")
    exit(1)
}

let markdown: String = {
    let path = CommandLine.arguments[1]

    do {
        let path = (path as NSString).expandingTildeInPath
        return try String(contentsOfFile: path)
    } catch {
        print("""
        🛑 Failed to open Markdown file at '\(path)':
        ---
        \(error.localizedDescription)
        ---
        """)
        exit(1)
    }
}()

let decorator = MarkdownDecorator()
print(decorator.decorate(markdown))
