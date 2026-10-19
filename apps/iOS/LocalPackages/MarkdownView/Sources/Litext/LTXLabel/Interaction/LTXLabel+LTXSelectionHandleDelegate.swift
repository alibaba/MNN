//
//  LTXLabel+LTXSelectionHandleDelegate.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/8/25.
//

import UIKit

extension LTXLabel: LTXSelectionHandleDelegate {
    func selectionHandleDidMove(_ type: LTXSelectionHandle.HandleType, toLocationInSuperView point: CGPoint) {
        guard let selectionRange, let textLocation = nearestTextIndexAtPoint(point) else { return }
        switch type {
        case .start:
            let startLocation = min(textLocation, selectionRange.location + selectionRange.length - 1)
            let length = selectionRange.location + selectionRange.length - startLocation
            self.selectionRange = .init(
                location: startLocation,
                length: length
            )
        case .end:
            let startLocation = selectionRange.location
            let endingLocation = max(textLocation, startLocation + 1)
            self.selectionRange = .init(
                location: startLocation,
                length: endingLocation - startLocation
            )
        }
    }
}
