//
//  LTXLabel+Rect.swift
//  Litext
//
//  Created by 秋星桥 on 3/27/25.
//

import Foundation

extension LTXLabel {
    func convertRectFromTextLayout(_ rect: CGRect, insetForInteraction useInset: Bool) -> CGRect {
        var result = rect
        result.origin.y = bounds.height - result.origin.y - result.size.height
        if useInset { result = result.insetBy(dx: -4, dy: -4) }
        return result
    }
}
