//
//  Created by ktiays on 2025/1/22.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import CoreText
import Foundation

public extension CTLine {
    func glyphRuns() -> [CTRun] {
        CTLineGetGlyphRuns(self) as! [CTRun]
    }
}
