/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

#if os(macOS)

import Foundation
import ImageIO

extension CGImage {
    func write(to url: URL) {
        let destination = CGImageDestinationCreateWithURL(url as CFURL, kUTTypePNG, 1, nil)!
        CGImageDestinationAddImage(destination, self, nil)
        CGImageDestinationFinalize(destination)
    }
}

#endif
