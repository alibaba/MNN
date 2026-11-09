/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

internal extension String {
    var isCapitalized: Bool {
        guard let firstCharacter = first.map(String.init) else {
            return false
        }

        return firstCharacter != firstCharacter.lowercased()
    }

    var startsWithLetter: Bool {
        guard let firstCharacter = first else {
            return false
        }

        return CharacterSet.letters.contains(firstCharacter)
    }
}
