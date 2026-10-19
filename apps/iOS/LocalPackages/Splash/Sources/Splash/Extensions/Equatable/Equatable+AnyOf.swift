/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

extension Equatable {
    func isAny(of candidates: Self...) -> Bool {
        return candidates.contains(self)
    }

    func isAny<S: Sequence>(of candidates: S) -> Bool where S.Element == Self {
        return candidates.contains(self)
    }
}
