/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

/// Protocol used to define the grammar of a language to use for
/// syntax highlighting. See `SwiftGrammar` for a default implementation
/// of the Swift language grammar.
public protocol Grammar {
    /// The set of characters that make up the delimiters that separates
    /// tokens within the language, such as punctuation characters. You
    /// can control whether delimiters should be merged when forming
    /// tokens by implementing the `isDelimiter(mergableWith:)` method.
    var delimiters: CharacterSet { get }
    /// The rules that define the syntax of the language. When tokenizing,
    /// the rules will be iterated over in sequence, and the first rule
    /// that matches a given code segment will be used to determine that
    /// segment's token type.
    var syntaxRules: [SyntaxRule] { get }

    /// Return whether two delimiters should be merged into a single
    /// token, or whether they should be treated as separate ones.
    /// The delimiters are passed in the order in which they appear
    /// in the source code to be highlighted.
    /// - Parameter delimiterA: The first delimiter
    /// - Parameter delimiterB: The second delimiter
    func isDelimiter(_ delimiterA: Character,
                     mergableWith delimiterB: Character) -> Bool
}

public extension Grammar {
    func isDelimiter(_ delimiterA: Character,
                     mergableWith delimiterB: Character) -> Bool {
        return true
    }
}
