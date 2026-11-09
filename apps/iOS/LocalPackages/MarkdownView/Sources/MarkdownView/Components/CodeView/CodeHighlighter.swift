//
//  Created by ktiays on 2025/1/22.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import Foundation
import LRUCache
import OrderedCollections
import Splash
import UIKit

private let kMaxCacheSize = 64 // for each language
private let kPrefixLength = 8

public final class CodeHighlighter {
    public typealias HighlightMap = [NSRange: UIColor]

    public private(set) var renderCache = LRUCache<Int, HighlightMap>(countLimit: 256)

    private init() {}
    public static let current = CodeHighlighter()
}

public extension CodeHighlighter {
    func key(for content: String, language: String?) -> Int {
        var hasher = Hasher()
        hasher.combine(content)
        hasher.combine(language?.lowercased() ?? "")
        return hasher.finalize()
    }

    func highlight(
        key: Int?,
        content: String,
        language: String?,
        theme: MarkdownTheme = .default // doesn't matter we use color only
    ) -> [NSRange: UIColor] {
        let key = key ?? self.key(for: content, language: language)
        if let value = renderCache.value(forKey: key) {
            return value
        }
        let highlightedAttributeString = highlightedAttributeString(
            language: language ?? "",
            content: content,
            theme: theme
        )
        let map = extractColorAttributes(from: highlightedAttributeString)
        renderCache.setValue(map, forKey: key)
        return map
    }
}

private extension CodeHighlighter {
    func highlightedAttributeString(language: String, content: String, theme: MarkdownTheme) -> NSAttributedString {
        let codeTheme = theme.codeTheme(withFont: theme.fonts.code)
        let format = AttributedStringOutputFormat(theme: codeTheme)
        let base = {
            switch language.lowercased() {
            case "text", "plaintext":
                return NSAttributedString(string: content)
            case "swift":
                let splash = SyntaxHighlighter(format: format, grammar: SwiftGrammar())
                return splash.highlight(content)
            default:
                let splash = SyntaxHighlighter(format: format)
                return splash.highlight(content)
            }
        }()
        guard let finalizer = base.mutableCopy() as? NSMutableAttributedString else {
            return .init()
        }
        finalizer.addAttributes([
            .font: codeTheme.font,
        ], range: .init(location: 0, length: finalizer.length))
        return finalizer
    }

    func extractColorAttributes(from attributedString: NSAttributedString) -> HighlightMap {
        var attributes: [NSRange: UIColor] = [:]

        attributedString.enumerateAttribute(
            .foregroundColor,
            in: NSRange(location: 0, length: attributedString.length)
        ) { value, range, _ in
            guard let color = value as? UIColor else { return }
            attributes[range] = color
        }

        return attributes
    }
}

public extension CodeHighlighter.HighlightMap {
    func apply(to content: String, with theme: MarkdownTheme) -> NSMutableAttributedString {
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = CodeViewConfiguration.codeLineSpacing

        let plainTextColor = theme.colors.code
        let attributedContent: NSMutableAttributedString = .init(
            string: content,
            attributes: [
                .font: theme.fonts.code,
                .paragraphStyle: paragraphStyle,
                .foregroundColor: plainTextColor,
            ]
        )

        let length = attributedContent.length
        for (range, color) in self {
            guard range.location >= 0, range.upperBound <= length else { continue }
            guard color != plainTextColor else { continue }
            attributedContent.addAttributes([.foregroundColor: color], range: range)
        }
        return attributedContent
    }
}
