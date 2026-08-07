//
//  Created by ktiays on 2025/1/20.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import CoreText
import Litext
import UIKit

// MARK: - TextBuilder Callback Types

extension TextBuilder {
    typealias DrawingCallback = (CGContext, CTLine, CGPoint) -> Void
    typealias BulletDrawingCallback = (CGContext, CTLine, CGPoint, Int) -> Void
    typealias CheckboxDrawingCallback = (CGContext, CTLine, CGPoint, Bool) -> Void
    typealias NumberedDrawingCallback = (CGContext, CTLine, CGPoint, Int) -> Void
    typealias BlockquoteMarkingCallback = (CGContext, CTLine, CGPoint) -> Void
    typealias BlockquoteDrawingCallback = (CGContext, CTLine, CGPoint) -> Void
}

// MARK: - RenderText

struct RenderText {
    let attributedString: NSAttributedString
    let fullWidthAttachments: [LTXAttachment]
}

// MARK: - String Extension

extension String {
    func deletingSuffix(of characterSet: CharacterSet) -> String {
        var result = self
        while let lastChar = result.last, characterSet.contains(lastChar.unicodeScalars.first!) {
            result.removeLast()
        }
        return result
    }
}
