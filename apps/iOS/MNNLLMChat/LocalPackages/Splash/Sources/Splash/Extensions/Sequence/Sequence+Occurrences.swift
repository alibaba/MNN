/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

internal extension Sequence where Element: Equatable {
    func numberOfOccurrences(of target: Element) -> Int {
        return reduce(0) { count, element in
            return element == target ? count + 1 : count
        }
    }
}
