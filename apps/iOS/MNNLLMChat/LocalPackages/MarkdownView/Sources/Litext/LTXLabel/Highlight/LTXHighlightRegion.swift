//
//  Created by Lakr233 & Helixform on 2025/2/18.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import CoreGraphics
import Foundation

public class LTXHighlightRegion {
    public private(set) var rects: [NSValue] = []
    public private(set) var attributes: [NSAttributedString.Key: Any]
    public private(set) var stringRange: NSRange

    var associatedObject: AnyObject?

    init(attributes: [NSAttributedString.Key: Any], stringRange: NSRange) {
        self.attributes = attributes
        self.stringRange = stringRange
    }

    func addRect(_ rect: CGRect) {
        rects.append(NSValue(cgRect: rect))
    }
}
