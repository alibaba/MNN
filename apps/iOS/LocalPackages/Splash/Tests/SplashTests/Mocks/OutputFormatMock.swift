/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation
import Splash

struct OutputFormatMock: OutputFormat {
    let builder: OutputBuilderMock

    func makeBuilder() -> OutputBuilderMock {
        return builder
    }
}
