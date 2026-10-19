//
//  Created by ktiays on 2025/1/22.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import Foundation

extension CFRange {
    var nsRange: NSRange {
        NSMakeRange(location == kCFNotFound ? NSNotFound : location, length)
    }
}
