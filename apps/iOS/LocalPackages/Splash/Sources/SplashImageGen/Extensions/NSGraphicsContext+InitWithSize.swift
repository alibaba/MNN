/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

#if os(macOS)

import Cocoa

extension NSGraphicsContext {
    convenience init(size: CGSize) {
        let scale: CGFloat = 2

        let context = CGContext(
            data: nil,
            width: Int(size.width * scale),
            height: Int(size.height * scale),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        )!

        context.scaleBy(x: scale, y: scale)

        self.init(cgContext: context, flipped: false)
    }
}

#endif
