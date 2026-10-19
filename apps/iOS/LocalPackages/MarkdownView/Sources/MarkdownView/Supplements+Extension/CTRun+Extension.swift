//
//  Created by ktiays on 2025/1/22.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import CoreText
import UIKit

public extension CTRun {
    var attributes: [NSAttributedString.Key: Any] {
        (CTRunGetAttributes(self) as NSDictionary as! [String: Any])
            .reduce([:]) { (partialResult: [NSAttributedString.Key: Any], tuple: (key: String, value: Any)) in
                var result = partialResult
                let attributeName = NSAttributedString.Key(rawValue: tuple.key)
                result[attributeName] = tuple.value
                return result
            }
    }
}
