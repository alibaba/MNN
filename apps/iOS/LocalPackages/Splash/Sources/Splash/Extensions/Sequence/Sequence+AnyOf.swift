/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

internal extension Sequence where Element: Equatable {
    func contains(anyOf candidates: Element...) -> Bool {
        return contains(anyOf: candidates)
    }

    func contains<S: Sequence>(anyOf candidates: S) -> Bool where S.Element == Element {
        for candidate in candidates {
            if contains(candidate) {
                return true
            }
        }

        return false
    }
}
