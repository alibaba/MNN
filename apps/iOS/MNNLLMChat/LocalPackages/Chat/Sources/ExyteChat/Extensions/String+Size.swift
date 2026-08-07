//
//  Created by Alex.M on 08.07.2022.
//

import SwiftUI
import UIKit

extension String {

    static func localizedFormat(key: String, _ args: CVarArg...) -> String {
        let localizedString = NSLocalizedString(key, comment: "")
        return String(format: localizedString, arguments: args)
    }

    static var markdownOptions = AttributedString.MarkdownParsingOptions(
        allowsExtendedAttributes: false,
        interpretedSyntax: .inlineOnlyPreservingWhitespace,
        failurePolicy: .returnPartiallyParsedIfPossible,
        languageCode: nil
    )

    func width(withConstrainedWidth width: CGFloat, font: UIFont, messageUseMarkdown: Bool) -> CGFloat {
        let constraintRect = CGSize(width: width, height: .greatestFiniteMagnitude)
        let boundingBox = toAttrString(font: font, messageUseMarkdown: messageUseMarkdown).boundingRect(with: constraintRect,
                                                      options: .usesLineFragmentOrigin,
                                                      context: nil)

        return ceil(boundingBox.width)
    }

    func toAttrString(font: UIFont, messageUseMarkdown: Bool) -> NSAttributedString {
        var str = messageUseMarkdown ? (try? AttributedString(markdown: self, options: String.markdownOptions)) ?? AttributedString(self) : AttributedString(self)
        str.setAttributes(AttributeContainer([.font: font]))
        return NSAttributedString(str)
    }

    public func lastLineWidth(labelWidth: CGFloat, font: UIFont, messageUseMarkdown: Bool) -> CGFloat {
        // Create instances of NSLayoutManager, NSTextContainer and NSTextStorage
        let attrString = toAttrString(font: font, messageUseMarkdown: messageUseMarkdown)
        let availableSize = CGSize(width: labelWidth, height: .infinity)
        let layoutManager = NSLayoutManager()
        let textContainer = NSTextContainer(size: availableSize)
        let textStorage = NSTextStorage(attributedString: attrString)

        // Configure layoutManager and textStorage
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)

        // Configure textContainer
        textContainer.lineFragmentPadding = 0.0
        textContainer.lineBreakMode = .byWordWrapping
        textContainer.maximumNumberOfLines = 0

        let lastGlyphIndex = layoutManager.glyphIndexForCharacter(at: attrString.length - 1)
        let lastLineFragmentRect = layoutManager.lineFragmentUsedRect(
            forGlyphAt: lastGlyphIndex,
            effectiveRange: nil)

        return lastLineFragmentRect.maxX
    }

    func numberOfLines(labelWidth: CGFloat, font: UIFont, messageUseMarkdown: Bool) -> Int {
        let attrString = toAttrString(font: font, messageUseMarkdown: messageUseMarkdown)
        let availableSize = CGSize(width: labelWidth, height: .infinity)
        let textSize = attrString.boundingRect(with: availableSize, options: .usesLineFragmentOrigin, context: nil)
        let lineHeight = font.lineHeight
        return Int(ceil(textSize.height/lineHeight))
    }
}
