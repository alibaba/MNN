/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

/// Protocol used to define a builder for a highlighted string that's
/// returned as output from `SyntaxHighlighter`. Each builder defines
/// its own output type through the `Output` associated type, and can
/// add the various tokens and other text found in the highlighted code
/// in whichever fashion it wants.
public protocol OutputBuilder {
    /// The type of output that this builder produces
    associatedtype Output
    /// Add a token with a given type to the builder
    mutating func addToken(_ token: String, ofType type: TokenType)
    /// Add some plain text, without any formatting, to the builder
    mutating func addPlainText(_ text: String)
    /// Add some whitespace to the builder
    mutating func addWhitespace(_ whitespace: String)
    /// Build the final output based on the builder's current state
    mutating func build() -> Output
}
