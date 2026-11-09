//
//  TextBuilder+Do.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/9/25.
//

import CoreText
import Foundation
import UIKit

private func builtinSystemImage(_ name: String, size: CGFloat = 16) -> UIImage {
    guard let image = UIImage(
        systemName: name,
        withConfiguration: UIImage.SymbolConfiguration(scale: .small)
    ) else { return .init() }
    let templateImage = image.withTintColor(.label, renderingMode: .alwaysTemplate)
    return templateImage.resized(to: .init(width: size, height: size))
}

private let kCheckedBoxImage = builtinSystemImage("checkmark.square.fill")
private let kUncheckedBoxImage = builtinSystemImage("square")

private func kNumberCircleImage(_ number: Int) -> UIImage {
    builtinSystemImage("\(number).circle.fill")
}

extension TextBuilder {
    @inline(__always)
    static func lineBoundingBox(_ line: CTLine, lineOrigin: CGPoint) -> CGRect {
        var ascent: CGFloat = 0
        var descent: CGFloat = 0
        let width = CTLineGetTypographicBounds(line, &ascent, &descent, nil)
        return .init(x: lineOrigin.x, y: lineOrigin.y - descent, width: width, height: ascent + descent)
    }

    static func build(view: MarkdownTextView, viewProvider: ReusableViewProvider) -> BuildResult {
        let context: MarkdownTextView.PreprocessedContent = view.document
        let theme: MarkdownTheme = view.theme

        var blockquoteMarkingStorage: CGFloat? = nil

        @discardableResult
        func populateContextColorFromFirstRun(context: CGContext, line: CTLine) -> UIColor {
            var textColor = theme.colors.body
            if let firstRun = line.glyphRuns().first,
               let attributes = CTRunGetAttributes(firstRun) as? [NSAttributedString.Key: Any],
               let color = attributes[.foregroundColor] as? UIColor
            {
                textColor = color
            }
            context.setStrokeColor(textColor.cgColor)
            context.setFillColor(textColor.cgColor)
            return textColor
        }

        return TextBuilder(nodes: context.blocks, context: context, viewProvider: viewProvider)
            .withTheme(theme)
            .withBulletDrawing { context, line, lineOrigin, depth in
                let radius: CGFloat = 3
                let boundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)
                populateContextColorFromFirstRun(context: context, line: line)
                let rect = CGRect(
                    x: boundingBox.minX - 16,
                    y: boundingBox.midY - radius,
                    width: radius * 2,
                    height: radius * 2
                )
                if depth == 0 {
                    context.fillEllipse(in: rect)
                } else if depth == 1 {
                    context.strokeEllipse(in: rect)
                } else {
                    context.fill(rect)
                }
            }
            .withNumberedDrawing { context, line, lineOrigin, num in
                let rect = lineBoundingBox(line, lineOrigin: lineOrigin)
                    .offsetBy(dx: -16, dy: 0)
                    .offsetBy(dx: -8, dy: 0)
                let image = kNumberCircleImage(num)
                guard let cgImage = image.cgImage else { return }
                let imageSize = image.size
                let targetRect: CGRect = .init(
                    x: rect.minX,
                    y: rect.midY - imageSize.height / 2,
                    width: imageSize.width,
                    height: imageSize.height
                )
                let textColor = populateContextColorFromFirstRun(context: context, line: line)
                context.clip(to: targetRect, mask: cgImage)
                context.setFillColor(textColor.cgColor)
                context.fill(targetRect)
            }
            .withCheckboxDrawing { context, line, lineOrigin, isChecked in
                let rect = lineBoundingBox(line, lineOrigin: lineOrigin)
                    .offsetBy(dx: -16, dy: 0)
                    .offsetBy(dx: -8, dy: 0)
                let image = if isChecked { kCheckedBoxImage } else { kUncheckedBoxImage }
                guard let cgImage = image.cgImage else { return }
                let imageSize = image.size
                let targetRect: CGRect = .init(
                    x: rect.minX,
                    y: rect.midY - imageSize.height / 2,
                    width: imageSize.width,
                    height: imageSize.height
                )
                let textColor = populateContextColorFromFirstRun(context: context, line: line)
                context.clip(to: targetRect, mask: cgImage)
                context.setFillColor(textColor.withAlphaComponent(0.24).cgColor)
                context.fill(targetRect)
            }
            .withThematicBreakDrawing { [weak view] context, line, lineOrigin in
                guard let view else { return }
                let boundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)

                context.setLineWidth(1)
                context.setStrokeColor(UIColor.label.withAlphaComponent(0.1).cgColor)
                context.move(to: .init(x: boundingBox.minX, y: boundingBox.midY))
                context.addLine(to: .init(x: boundingBox.minX + view.bounds.width, y: boundingBox.midY))
                context.strokePath()
            }
            .withCodeDrawing { [weak view] _, line, lineOrigin in
                guard let view else { return }
                guard let firstRun = line.glyphRuns().first else { return }
                let attributes = firstRun.attributes
                guard let codeView = attributes[.contextView] as? CodeView else {
                    assertionFailure()
                    return
                }

                if codeView.superview != view { view.addSubview(codeView) }
                let intrinsicContentSize = codeView.intrinsicContentSize
                let lineBoundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)
                var leftIndent: CGFloat = 0
                if let paragraphStyle = attributes[.paragraphStyle] as? NSParagraphStyle {
                    leftIndent = paragraphStyle.headIndent
                }

                codeView.frame = .init(
                    origin: .init(x: lineOrigin.x + leftIndent, y: view.bounds.height - lineBoundingBox.maxY),
                    size: .init(width: view.bounds.width - leftIndent, height: intrinsicContentSize.height)
                )
                codeView.previewAction = view.codePreviewHandler
            }
            .withTableDrawing { [weak view] _, line, lineOrigin in
                guard let view else { return }
                guard let firstRun = line.glyphRuns().first else { return }
                let attributes = firstRun.attributes
                guard let tableView = attributes[.contextView] as? TableView else {
                    assertionFailure()
                    return
                }

                if tableView.superview != view { view.addSubview(tableView) }
                let lineBoundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)
                let intrinsicContentSize = tableView.intrinsicContentSize
                var leftIndent: CGFloat = 0
                if let paragraphStyle = attributes[.paragraphStyle] as? NSParagraphStyle {
                    leftIndent = paragraphStyle.headIndent
                }

                tableView.frame = .init(
                    x: lineOrigin.x + leftIndent,
                    y: view.bounds.height - lineBoundingBox.maxY,
                    width: view.bounds.width - leftIndent,
                    height: intrinsicContentSize.height
                )
            }
            .withBlockquoteMarking { _, line, lineOrigin in
                let boundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)
                blockquoteMarkingStorage = boundingBox.maxY
            }
            .withBlockquoteDrawing { context, line, lineOrigin in
                let boundingBox = lineBoundingBox(line, lineOrigin: lineOrigin)
                defer { blockquoteMarkingStorage = nil }
                let quotingLineHeight: CGFloat = blockquoteMarkingStorage! - boundingBox.minY
                let lineRect = CGRect(
                    x: 0,
                    y: blockquoteMarkingStorage! - quotingLineHeight,
                    width: 4,
                    height: quotingLineHeight
                )
                context.setFillColor(theme.colors.body.withAlphaComponent(0.1).cgColor)
                let roundedPath = CGPath(roundedRect: lineRect, cornerWidth: 2, cornerHeight: 2, transform: nil)
                context.addPath(roundedPath)
                context.fillPath()
            }
            .build()
    }
}
