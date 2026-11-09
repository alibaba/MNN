/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

/// Protocol used to define an output format for a `SyntaxHighlighter`.
/// Default implementations of this protocol are provided for HTML and
/// NSAttributedString outputs, and custom ones can be defined by
/// conforming to this protocol and passing the implementation to a
/// syntax highlighter when it's created.
public protocol OutputFormat {
    /// The type of builder that this output format uses. The builder's
    /// `Output` type determines the output type of the format.
    associatedtype Builder: OutputBuilder

    /// Make a new instance of the output format's builder. This will be
    /// called once per syntax highlighting session. The builder is expected
    /// to be a newly created, blank instance.
    func makeBuilder() -> Builder
}
