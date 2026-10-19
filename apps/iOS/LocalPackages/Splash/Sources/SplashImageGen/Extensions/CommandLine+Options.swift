/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

#if os(macOS)

import Foundation
import Splash

extension CommandLine {
    struct Options {
        let code: String
        let outputURL: URL
        let padding: CGFloat
        let font: Font
    }

    static func makeOptions() -> Options? {
        guard arguments.count > 2 else {
            return nil
        }

        let defaults = UserDefaults.standard

        return Options(
            code: arguments[1],
            outputURL: resolveOutputURL(),
            padding: CGFloat(defaults.int(forKey: "p", default: 20)),
            font: resolveFont(from: defaults)
        )
    }

    private static func resolveOutputURL() -> URL {
        let path = arguments[2] as NSString
        return URL(fileURLWithPath: path.expandingTildeInPath)
    }

    private static func resolveFont(from defaults: UserDefaults) -> Font {
        let size = Double(defaults.int(forKey: "s", default: 20))

        guard let path = defaults.string(forKey: "f") else {
            return Font(size: size)
        }

        return Font(path: path, size: size)
    }
}

private extension UserDefaults {
    func int(forKey key: String, default: CGFloat) -> CGFloat {
        guard value(forKey: key) != nil else {
            return `default`
        }

        return CGFloat(integer(forKey: key))
    }
}

#endif
