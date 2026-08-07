/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

/// Protocol used to define syntax rules for a language `Grammar`.
/// Each rule is associated with a certain `TokenType` and, when
/// evaluated, is asked to check whether it matches a given segment
/// of code. If the rule matches then the rule's token type will be
/// associated with the given segment's current token.
public protocol SyntaxRule {
    /// The token type that this syntax rule represents
    var tokenType: TokenType { get }

    /// Determine if the syntax rule matches a given segment. If it's
    /// a match, then the rule's `tokenType` will be associated with
    /// the segment's current token.
    func matches(_ segment: Segment) -> Bool
}
