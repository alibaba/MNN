//
//  MarkdownParser.swift
//  FlowMarkdownView
//
//  Created by 秋星桥 on 2025/1/2.
//

import cmark_gfm
import cmark_gfm_extensions
import Foundation

public class MarkdownParser {
    public init() {}

    func withParser<T>(_ block: (UnsafeMutablePointer<cmark_parser>) -> T) -> T {
        let parser = cmark_parser_new(CMARK_OPT_DEFAULT)!
        cmark_gfm_core_extensions_ensure_registered()
        let extensionNames = [
            "autolink",
            "strikethrough",
            "tagfilter",
            "tasklist",
            "table",
        ]
        for extensionName in extensionNames {
            guard let syntaxExtension = cmark_find_syntax_extension(extensionName) else {
                assertionFailure()
                continue
            }
            cmark_parser_attach_syntax_extension(parser, syntaxExtension)
        }
        defer { cmark_parser_free(parser) }
        return block(parser)
    }

    public struct ParseResult {
        public let document: [MarkdownBlockNode]
        public let mathContext: [Int: String]
    }

    public func parse(_ markdown: String) -> ParseResult {
        let math = MathContext(preprocessText: markdown)
        math.process()
        let markdown = math.indexedContent ?? markdown
        let nodes = withParser { parser in
            markdown.withCString { str in
                cmark_parser_feed(parser, str, strlen(str))
                return cmark_parser_finish(parser)
            }
        }
        var blocks = dumpBlocks(root: nodes)
        blocks = finalizeMathBlocks(blocks, mathContext: math)
        return .init(document: blocks, mathContext: math.contents)
    }

    public struct RootBlockRange {
        public let type: MarkdownNodeType
        public let startIndex: String.Index
        public let endIndex: String.Index
    }

    public func parseBlockRange(_ markdown: String) -> [RootBlockRange] {
        var ranges = [RootBlockRange]()

        let root = withParser { parser in
            markdown.withCString { str in
                cmark_parser_feed(parser, str, strlen(str))
                return cmark_parser_finish(parser)
            }
        }
        guard let root else {
            assertionFailure()
            return ranges
        }

        assert(root.pointee.type == CMARK_NODE_DOCUMENT.rawValue)
        for block in root.children {
            let node = block.pointee

            let startLine = Int(node.start_line)
            let startColumn = Int(node.start_column)
            let endLine = Int(node.end_line)
            let endColumn = Int(node.end_column)

            guard let startIndex = getIndex(forLine: startLine, column: startColumn, in: markdown),
                  let endIndex = getIndex(forLine: endLine, column: endColumn, in: markdown)
            else {
                assertionFailure()
                continue
            }
            let content = RootBlockRange(type: block.nodeType, startIndex: startIndex, endIndex: endIndex)
            ranges.append(content)

            print("[*] block: \(block.nodeType) at \(content.startIndex) - \(content.endIndex)")
        }
        return ranges
    }
}

private func getIndex(forLine targetLine: Int, column targetColumn: Int, in text: String) -> String.Index? {
    var currentLine = 1
    var lineStartIndex = text.startIndex

    while currentLine < targetLine {
        guard let newlineIndex = text[lineStartIndex...].firstIndex(of: "\n") else {
            return nil
        }
        lineStartIndex = text.index(after: newlineIndex)
        currentLine += 1
    }

    // cmark 使用 1-based 列号，需要减1转换为 0-based
    let targetOffset = targetColumn - 1

    let lineEndIndex: String.Index = if let newlineIndex = text[lineStartIndex...].firstIndex(of: "\n") {
        newlineIndex
    } else {
        text.endIndex
    }

    let maxOffset = text.distance(from: lineStartIndex, to: lineEndIndex)

    if targetOffset > maxOffset {
        return lineEndIndex
    }

    if targetOffset < 0 {
        return lineStartIndex
    }

    return text.index(lineStartIndex, offsetBy: targetOffset)
}
