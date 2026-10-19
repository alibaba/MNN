/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

#if !os(Linux)

/// A theme describes what fonts and colors to use when rendering
/// certain output formats - such as `NSAttributedString`. Several
/// default implementations are provided - see Theme+Defaults.swift.
public struct Theme {
    /// What font to use to render the highlighted text
    public var font: Font
    /// What color to use for plain text (no highlighting)
    public var plainTextColor: Color
    /// What color to use for the background
    public var backgroundColor: Color
    /// What color to use for the text's highlighted tokens
    public var tokenColors: [TokenType: Color]

    public init(font: Font,
                plainTextColor: Color,
                tokenColors: [TokenType: Color],
                backgroundColor: Color = Color(white: 0.12, alpha: 1)) {
        self.font = font
        self.plainTextColor = plainTextColor
        self.tokenColors = tokenColors
        self.backgroundColor = backgroundColor
    }
}

#endif
