import Foundation

public extension Sequence<MarkdownInlineNode> {
    func collect<Result>(_ c: (MarkdownInlineNode) throws -> [Result]) rethrows -> [Result] {
        try flatMap { try $0.collect(c) }
    }
}

public extension MarkdownInlineNode {
    func collect<Result>(_ c: (MarkdownInlineNode) throws -> [Result]) rethrows -> [Result] {
        try children.collect(c) + c(self)
    }
}
