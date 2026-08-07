//
//  Created by Litext Team.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import CoreGraphics
import CoreText
import Foundation
import QuartzCore

private let kDeduplicateSelectionNotification = Notification.Name(
    rawValue: "LTXLabelDeduplicateSelectionNotification"
)

extension LTXLabel {
    func updateSelectionLayer() {
        selectionLayer?.removeFromSuperlayer()
        selectionLayer = nil

        #if canImport(UIKit) && !targetEnvironment(macCatalyst)
            selectionHandleStart.isHidden = true
            selectionHandleEnd.isHidden = true
        #endif

        guard let range = selectionRange,
              range.location != NSNotFound,
              range.length > 0
        else {
            #if canImport(UIKit) && !targetEnvironment(macCatalyst)
                hideSelectionMenuController()
            #endif
            return
        }

        let selectionPath = LTXPlatformBezierPath()
        let selectionRects = textLayout.rects(for: range)
        guard !selectionRects.isEmpty else {
            #if canImport(UIKit) && !targetEnvironment(macCatalyst)
                hideSelectionMenuController()
            #endif
            return
        }

        createSelectionPath(selectionPath, fromRects: selectionRects)
        createSelectionLayer(withPath: selectionPath)

        #if canImport(UIKit) && !targetEnvironment(macCatalyst)
            showSelectionMenuController()
        #endif

        #if canImport(UIKit) && !targetEnvironment(macCatalyst)
            selectionHandleStart.isHidden = false
            selectionHandleEnd.isHidden = false

            var beginRect = textLayout.rects(
                for: NSRange(location: range.location, length: 1)
            ).first ?? .zero
            beginRect = convertRectFromTextLayout(beginRect, insetForInteraction: false)
            selectionHandleStart.frame = .init(
                x: beginRect.minX - LTXSelectionHandle.knobRadius - 1,
                y: beginRect.minY - LTXSelectionHandle.knobRadius,
                width: LTXSelectionHandle.knobRadius * 2,
                height: beginRect.height + LTXSelectionHandle.knobRadius
            )
            var endRect = textLayout.rects(
                for: NSRange(location: range.location + range.length - 1, length: 1)
            ).first ?? .zero
            endRect = convertRectFromTextLayout(endRect, insetForInteraction: false)
            selectionHandleEnd.frame = .init(
                x: endRect.maxX - LTXSelectionHandle.knobRadius + 1,
                y: endRect.minY,
                width: LTXSelectionHandle.knobRadius * 2,
                height: endRect.height + LTXSelectionHandle.knobRadius
            )
        #endif

        NotificationCenter.default.post(name: kDeduplicateSelectionNotification, object: self)
    }

    func registerNotificationCenterForSelectionDeduplicate() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(deduplicateSelection),
            name: kDeduplicateSelectionNotification,
            object: nil
        )
    }

    @objc private func deduplicateSelection(_ notification: Notification) {
        guard let object = notification.object as? LTXLabel, object != self else { return }
        clearSelection()
    }

    private func createSelectionPath(_ selectionPath: LTXPlatformBezierPath, fromRects rects: [CGRect]) {
        for rect in rects {
            let convertedRect = convertRectFromTextLayout(rect, insetForInteraction: false)

            #if canImport(UIKit)
                let subpath = LTXPlatformBezierPath(rect: convertedRect)
                selectionPath.append(subpath)
            #elseif canImport(AppKit)
                let subpath = LTXPlatformBezierPath(rect: convertedRect)
                selectionPath.appendPath(subpath)
            #endif
        }
    }

    private func createSelectionLayer(withPath path: LTXPlatformBezierPath) {
        let selLayer = CAShapeLayer()

        #if canImport(UIKit)
            selLayer.path = path.cgPath
        #elseif canImport(AppKit)
            selLayer.path = path.quartzPath
        #endif

        #if canImport(UIKit)
            selLayer.fillColor = UIColor.systemBlue.withAlphaComponent(0.1).cgColor
        #elseif canImport(AppKit)
            selLayer.fillColor = NSColor.linkColor.withAlphaComponent(0.1).cgColor
        #endif

        #if canImport(UIKit)
            layer.insertSublayer(selLayer, at: 0)
        #elseif canImport(AppKit)
            layer?.insertSublayer(selLayer, at: 0)
        #endif

        selectionLayer = selLayer
    }
}
