/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

#if os(macOS)

import Cocoa

extension NSGraphicsContext {
    func fill(with color: NSColor, in rect: CGRect) {
        cgContext.setFillColor(color.cgColor)
        cgContext.fill(rect)
    }
}

#endif
