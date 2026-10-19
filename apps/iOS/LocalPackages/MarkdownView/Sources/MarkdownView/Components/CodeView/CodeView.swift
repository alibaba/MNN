//
//  Created by ktiays on 2025/1/22.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import Litext
import UIKit

final class CodeView: UIView {
    // MARK: - CONTENT

    ///
    /// **PLEASE STRICTLY FOLLOW THE ORDER AS IS**
    /// theme -> language -> highlighMap -> content
    ///

    var theme: MarkdownTheme = .default {
        didSet {
            languageLabel.font = theme.fonts.code
            updateLineNumberView()
        }
    }

    var language: String = "" {
        didSet {
            languageLabel.text = language.isEmpty ? "</>" : language
        }
    }

    var highlightMap: CodeHighlighter.HighlightMap = .init()

    var content: String = "" {
        didSet {
            guard oldValue != content else { return }
            textView.attributedText = highlightMap.apply(to: content, with: theme)
            lineNumberView.updateForContent(content)
            updateLineNumberView()
        }
    }

    // MARK: CONTENT -

    var previewAction: ((String?, NSAttributedString) -> Void)? {
        didSet {
            setNeedsLayout()
        }
    }

    private let callerIdentifier = UUID()
    private var currentTaskIdentifier: UUID?

    lazy var barView: UIView = .init()
    lazy var scrollView: UIScrollView = .init()
    lazy var languageLabel: UILabel = .init()
    lazy var textView: LTXLabel = .init()
    lazy var copyButton: UIButton = .init()
    lazy var previewButton: UIButton = .init()
    lazy var lineNumberView: LineNumberView = .init()

    override init(frame: CGRect) {
        super.init(frame: frame)
        configureSubviews()
        updateLineNumberView()
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    static func intrinsicHeight(for content: String, theme: MarkdownTheme = .default) -> CGFloat {
        CodeViewConfiguration.intrinsicHeight(for: content, theme: theme)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        performLayout()
        updateLineNumberView()
    }

    override var intrinsicContentSize: CGSize {
        let labelSize = languageLabel.intrinsicContentSize
        let barHeight = labelSize.height + CodeViewConfiguration.barPadding * 2
        let textSize = textView.intrinsicContentSize
        let supposedHeight = Self.intrinsicHeight(for: content, theme: theme)

        let lineNumberWidth = lineNumberView.intrinsicContentSize.width

        return CGSize(
            width: max(
                labelSize.width + CodeViewConfiguration.barPadding * 2,
                lineNumberWidth + textSize.width + CodeViewConfiguration.codePadding * 2
            ),
            height: max(
                barHeight + textSize.height + CodeViewConfiguration.codePadding * 2,
                supposedHeight
            )
        )
    }

    @objc func handleCopy(_: UIButton) {
        UIPasteboard.general.string = content
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    @objc func handlePreview(_: UIButton) {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        previewAction?(language, textView.attributedText)
    }

    func updateLineNumberView() {
        let font = theme.fonts.code
        lineNumberView.textColor = theme.colors.body.withAlphaComponent(0.5)

        let lineCount = max(content.components(separatedBy: .newlines).count, 1)
        let textViewContentHeight = textView.intrinsicContentSize.height

        lineNumberView.configure(
            lineCount: lineCount,
            contentHeight: textViewContentHeight,
            font: font,
            textColor: .secondaryLabel
        )

        lineNumberView.padding = UIEdgeInsets(
            top: CodeViewConfiguration.codePadding,
            left: CodeViewConfiguration.lineNumberPadding,
            bottom: CodeViewConfiguration.codePadding,
            right: CodeViewConfiguration.lineNumberPadding
        )
    }
}

extension CodeView: LTXAttributeStringRepresentable {
    func attributedStringRepresentation() -> NSAttributedString {
        textView.attributedText
    }
}
