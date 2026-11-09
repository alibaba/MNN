/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

internal extension CharacterSet {
    func contains(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else {
            return false
        }

        return contains(scalar)
    }
}
