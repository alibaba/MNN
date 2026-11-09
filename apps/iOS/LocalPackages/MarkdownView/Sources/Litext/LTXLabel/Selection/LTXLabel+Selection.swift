//
//  Created by Lakr233 & Helixform on 2025/2/18.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import CoreGraphics
import CoreText
import Foundation
import QuartzCore

public extension LTXLabel {
    @objc func clearSelection() {
        selectionRange = nil
    }

    @discardableResult
    @objc func copySelectedText() -> NSAttributedString {
        guard let selectedText = selectedAttributedText() else {
            return .init()
        }

        #if canImport(UIKit)
            UIPasteboard.general.string = selectedText.string
        #elseif canImport(AppKit)
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString(selectedText.string, forType: .string)
        #endif

        return selectedText.copy() as! NSAttributedString
    }
}

extension LTXLabel {
    func updateSelectinoRange(withLocation location: CGPoint) {
        guard let startIndex = textLayout.nearestTextIndex(at: convertPointForTextLayout(interactionState.initialTouchLocation)),
              let endIndex = textLayout.nearestTextIndex(at: convertPointForTextLayout(location))
        else { return }
        selectionRange = NSRange(
            location: min(startIndex, endIndex),
            length: abs(endIndex - startIndex)
        )
    }

    func nearestTextIndexAtPoint(_ point: CGPoint) -> Int? {
        textLayout.nearestTextIndex(at: convertPointForTextLayout(point))
    }

    func textIndexAtPoint(_ point: CGPoint) -> Int? {
        textLayout.textIndex(at: convertPointForTextLayout(point))
    }

    func convertPointForTextLayout(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x, y: bounds.height - point.y)
    }

    public func isLocationInSelection(location: CGPoint) -> Bool {
        guard let range = selectionRange, range.length > 0 else { return false }
        let rects = textLayout.rects(for: range)
        return rects.map {
            convertRectFromTextLayout($0, insetForInteraction: true)
        }.contains { $0.contains(location) }
    }

    func selectedAttributedText() -> NSAttributedString? {
        guard let range = selectionRange,
              range.location != NSNotFound,
              range.length > 0,
              textLayout.attributedString.length > 0,
              range.location < textLayout.attributedString.length
        else {
            return nil
        }
        let maxLen = textLayout.attributedString.length - range.location

        let safeRange = NSRange(
            location: range.location,
            length: min(range.length, maxLen)
        )

        let selectedText = textLayout
            .attributedString
            .attributedSubstring(from: safeRange)

        let mutableResult = NSMutableAttributedString(attributedString: selectedText)
        mutableResult.enumerateAttribute(
            .ltxAttachment,
            in: NSRange(location: 0, length: mutableResult.length),
            options: []
        ) { value, range, _ in
            if let attachment = value as? LTXAttachment {
                mutableResult.replaceCharacters(
                    in: range,
                    with: attachment.attributedStringRepresentation()
                )
            }
        }

        return mutableResult
    }

    func selectedPlainText() -> String? {
        selectedAttributedText()?.string
    }
}
