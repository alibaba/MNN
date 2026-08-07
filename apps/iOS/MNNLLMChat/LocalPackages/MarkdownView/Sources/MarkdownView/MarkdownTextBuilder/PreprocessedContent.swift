//
//  PreprocessedContent.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/5/25.
//

import Foundation
import MarkdownParser

public extension MarkdownTextView {
    struct PreprocessedContent {
        public let blocks: [MarkdownBlockNode]
        public let rendered: RenderedTextContent.Map
        public let highlightMaps: [Int: CodeHighlighter.HighlightMap]

        public init(
            blocks: [MarkdownBlockNode],
            rendered: RenderedTextContent.Map,
            highlightMaps: [Int: CodeHighlighter.HighlightMap]
        ) {
            self.blocks = blocks
            self.rendered = rendered
            self.highlightMaps = highlightMaps
        }

        public init(parserResult: MarkdownParser.ParseResult, theme: MarkdownTheme) {
            blocks = parserResult.document
            rendered = parserResult.render(theme: theme)
            highlightMaps = parserResult.render(theme: theme)
        }

        public init() {
            blocks = .init()
            rendered = .init()
            highlightMaps = .init()
        }
    }
}

public extension MarkdownParser.ParseResult {
    fileprivate func renderMathContent(_ theme: MarkdownTheme, _ renderedContexts: inout [String: RenderedTextContent]) {
        for (key, value) in mathContext {
            let image = MathRenderer.renderToImage(
                latex: value,
                fontSize: theme.fonts.body.pointSize,
                textColor: theme.colors.body
            )?.withRenderingMode(.alwaysTemplate)
            let renderedContext = RenderedTextContent(
                image: image,
                text: value
            )
            let replacementText = MarkdownParser.replacementText(for: .math, identifier: .init(key))
            renderedContexts[replacementText] = renderedContext
        }
    }

    func render(theme: MarkdownTheme) -> RenderedTextContent.Map {
        var renderedContexts: [String: RenderedTextContent] = [:]
        renderMathContent(theme, &renderedContexts)
        return renderedContexts
    }
}

public extension MarkdownParser.ParseResult {
    fileprivate func renderHighlighMap(_: MarkdownTheme, highlightMaps: inout [Int: CodeHighlighter.HighlightMap]) {
        var iterator: [Any] = document
        while !iterator.isEmpty {
            let node = iterator.removeFirst()
            if let node = node as? MarkdownBlockNode {
                iterator.append(contentsOf: node.children)
                switch node {
                case let .blockquote(children):
                    iterator.append(contentsOf: children)
                case let .bulletedList(_, items):
                    iterator.append(contentsOf: items.flatMap(\.children))
                case let .numberedList(_, _, items):
                    iterator.append(contentsOf: items.flatMap(\.children))
                case let .taskList(_, items):
                    iterator.append(contentsOf: items.flatMap(\.children))
                case let .codeBlock(fenceInfo, content):
                    let key = CodeHighlighter.current.key(for: content, language: fenceInfo)
                    let map = CodeHighlighter.current.highlight(key: key, content: content, language: fenceInfo)
                    highlightMaps[key] = map
                case let .paragraph(content):
                    iterator.append(contentsOf: content)
                case let .heading(_, content):
                    iterator.append(contentsOf: content)
                case let .table(_, rows):
                    iterator.append(contentsOf: rows.flatMap(\.cells).map(\.content))
                case .thematicBreak:
                    break
                }
                continue
            }
            if let node = node as? MarkdownInlineNode {
                switch node {
                // 用户说这里很乱 不要高亮了
                // case let .code(string), let .html(string):
                // let key = CodeHighlighter.current.key(for: string, language: "")
                // let map = CodeHighlighter.current.highlight(key: key, content: string, language: "")
                // highlightMaps[key] = map
                default:
                    break
                }
                continue
            }
            continue
        }
    }

    func render(theme: MarkdownTheme) -> [Int: CodeHighlighter.HighlightMap] {
        var highlightMap = [Int: CodeHighlighter.HighlightMap]()
        renderHighlighMap(theme, highlightMaps: &highlightMap)
        return highlightMap
    }
}
