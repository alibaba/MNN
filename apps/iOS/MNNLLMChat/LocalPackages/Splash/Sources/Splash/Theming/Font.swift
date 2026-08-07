/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

#if !os(Linux)

/// A representation of a font, for use with a `Theme`.
/// Since Splash aims to be cross-platform, it uses this
/// simplified font representation rather than `NSFont`
/// or `UIFont`.
public struct Font {
    /// The underlying resource used to load the font
    public var resource: Resource
    /// The size (in points) of the font
    public var size: Double

    /// Initialize an instance with a path to a font file
    /// on disk and a size.
    public init(path: String, size: Double) {
        resource = .path((path as NSString).expandingTildeInPath)
        self.size = size
    }

    /// Initialize an instance with a size, and use an
    /// appropriate system font to render text.
    public init(size: Double) {
        resource = .system
        self.size = size
    }
}

public extension Font {
    /// Enum describing how to load the underlying resource for a font
    enum Resource {
        /// Use an appropriate system font
        case system
        /// Use a pre-loaded font
        case preloaded(Loaded)
        /// Load a font file from a given file system path
        case path(String)
    }
}

internal extension Font {
    func load() -> Loaded {
        switch resource {
        case .system:
            return loadDefaultFont()
        case .preloaded(let font):
            return font
        case .path(let path):
            return load(fromPath: path) ?? loadDefaultFont()
        }
    }

    private func loadDefaultFont() -> Loaded {
        let font: Loaded?

        #if os(iOS)
        font = UIFont(name: "Menlo-Regular", size: CGFloat(size))
        #else
        font = load(fromPath: "/Library/Fonts/Courier New.ttf")
        #endif

        return font ?? .systemFont(ofSize: CGFloat(size))
    }

    private func load(fromPath path: String) -> Loaded? {
        guard
            let url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, path as CFString, .cfurlposixPathStyle, false),
            let provider = CGDataProvider(url: url),
            let font = CGFont(provider)
        else {
            return nil
        }

        return CTFontCreateWithGraphicsFont(font, CGFloat(size), nil, nil)
    }
}

#endif

#if os(iOS)

import UIKit

public extension Font {
    typealias Loaded = UIFont
}

#elseif os(macOS)

import Cocoa

public extension Font {
    typealias Loaded = NSFont
}

#endif
