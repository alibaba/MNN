//
//  InlineNode+Render.swift
//  MarkdownView
//
//  Created by 秋星桥 on 2025/1/3.
//

import Foundation
import Litext
import MarkdownParser
import SwiftMath
import UIKit

extension [MarkdownInlineNode] {
    func render(theme: MarkdownTheme, context: MarkdownTextView.PreprocessedContent, viewProvider: ReusableViewProvider) -> NSMutableAttributedString {
        let result = NSMutableAttributedString()
        for node in self {
            result.append(node.render(theme: theme, context: context, viewProvider: viewProvider))
        }
        return result
    }
}

extension MarkdownInlineNode {
    func render(theme: MarkdownTheme, context: MarkdownTextView.PreprocessedContent, viewProvider: ReusableViewProvider) -> NSAttributedString {
        assert(Thread.isMainThread)
        switch self {
        case let .text(string):
            return NSMutableAttributedString(
                string: string,
                attributes: [
                    .font: theme.fonts.body,
                    .foregroundColor: theme.colors.body,
                ]
            )
        case .softBreak:
            return NSAttributedString(string: " ", attributes: [
                .font: theme.fonts.body,
                .foregroundColor: theme.colors.body,
            ])
        case .lineBreak:
            return NSAttributedString(string: "\n", attributes: [
                .font: theme.fonts.body,
                .foregroundColor: theme.colors.body,
            ])
        case let .code(string), let .html(string):
            let controlAttributes: [NSAttributedString.Key: Any] = [
                .font: theme.fonts.codeInline,
                .backgroundColor: theme.colors.codeBackground.withAlphaComponent(0.05),
            ]
            let text = NSMutableAttributedString(string: string, attributes: [.foregroundColor: theme.colors.code])
            text.addAttributes(controlAttributes, range: .init(location: 0, length: text.length))
            return text
        case let .emphasis(children):
            let ans = NSMutableAttributedString()
            children.map { $0.render(theme: theme, context: context, viewProvider: viewProvider) }.forEach { ans.append($0) }
            ans.addAttributes(
                [
                    .underlineStyle: NSUnderlineStyle.thick.rawValue,
                    .underlineColor: theme.colors.emphasis,
                ],
                range: NSRange(location: 0, length: ans.length)
            )
            return ans
        case let .strong(children):
            let ans = NSMutableAttributedString()
            children.map { $0.render(theme: theme, context: context, viewProvider: viewProvider) }.forEach { ans.append($0) }
            ans.addAttributes(
                [.font: theme.fonts.bold],
                range: NSRange(location: 0, length: ans.length)
            )
            return ans
        case let .strikethrough(children):
            let ans = NSMutableAttributedString()
            children.map { $0.render(theme: theme, context: context, viewProvider: viewProvider) }.forEach { ans.append($0) }
            ans.addAttributes(
                [.strikethroughStyle: NSUnderlineStyle.thick.rawValue],
                range: NSRange(location: 0, length: ans.length)
            )
            return ans
        case let .link(destination, children):
            let ans = NSMutableAttributedString()
            children.map { $0.render(theme: theme, context: context, viewProvider: viewProvider) }.forEach { ans.append($0) }
            ans.addAttributes(
                [
                    .link: destination,
                    .foregroundColor: theme.colors.highlight,
                ],
                range: NSRange(location: 0, length: ans.length)
            )
            return ans
        case let .image(source, _): // children => alternative text can be ignored?
            return NSAttributedString(
                string: source,
                attributes: [
                    .link: source,
                    .font: theme.fonts.body,
                    .foregroundColor: theme.colors.body,
                ]
            )
        case let .math(content, replacementIdentifier):
            if let item = context.rendered[replacementIdentifier], let image = item.image {
                var imageSize = image.size
                let lineHeight = theme.fonts.body.lineHeight

                if imageSize.height > lineHeight, imageSize.height < lineHeight * 1.5 {
                    // scale down for single-line equations
                    let aspectRatio = imageSize.width / imageSize.height
                    let scaledHeight = lineHeight
                    let scaledWidth = scaledHeight * aspectRatio
                    imageSize = CGSize(width: scaledWidth, height: scaledHeight)
                }

                let drawingCallback = LTXLineDrawingAction { context, line, lineOrigin in
                    let glyphRuns = CTLineGetGlyphRuns(line) as NSArray
                    var runOffsetX: CGFloat = 0
                    for i in 0 ..< glyphRuns.count {
                        let run = glyphRuns[i] as! CTRun
                        let attributes = CTRunGetAttributes(run) as! [NSAttributedString.Key: Any]
                        if attributes[.contextIdentifier] as? String == replacementIdentifier {
                            break
                        }
                        runOffsetX += CTRunGetTypographicBounds(run, CFRange(location: 0, length: 0), nil, nil, nil)
                    }

                    var ascent: CGFloat = 0
                    var descent: CGFloat = 0
                    CTLineGetTypographicBounds(line, &ascent, &descent, nil)
                    if imageSize.height > ascent { // we only draw above the line
                        let newWidth = imageSize.width * (ascent / imageSize.height)
                        imageSize = CGSize(width: newWidth, height: ascent)
                    }

                    let rect = CGRect(
                        x: lineOrigin.x + runOffsetX,
                        y: lineOrigin.y,
                        width: imageSize.width,
                        height: imageSize.height
                    )

                    context.saveGState()
                    context.translateBy(x: 0, y: rect.origin.y + rect.size.height)
                    context.scaleBy(x: 1, y: -1)
                    context.translateBy(x: 0, y: -rect.origin.y)
                    image.draw(in: rect)
                    context.restoreGState()
                }
                let attachment = LTXAttachment.hold(attrString: .init(string: content))
                attachment.size = imageSize
                return NSAttributedString(
                    string: LTXReplacementText,
                    attributes: [
                        LTXAttachmentAttributeName: attachment,
                        LTXLineDrawingCallbackName: drawingCallback,
                        kCTRunDelegateAttributeName as NSAttributedString.Key: attachment.runDelegate,
                        .contextIdentifier: replacementIdentifier,
                    ]
                )
            } else {
                return NSAttributedString(
                    string: content,
                    attributes: [
                        .font: theme.fonts.codeInline,
                        .foregroundColor: theme.colors.code,
                        .backgroundColor: theme.colors.codeBackground.withAlphaComponent(0.05),
                    ]
                )
            }
        }
    }
}
