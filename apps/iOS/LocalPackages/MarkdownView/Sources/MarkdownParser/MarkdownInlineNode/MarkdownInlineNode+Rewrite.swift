import Foundation

public extension Sequence<MarkdownInlineNode> {
    func rewrite(_ r: (MarkdownInlineNode) throws -> [MarkdownInlineNode]) rethrows -> [MarkdownInlineNode] {
        try flatMap { try $0.rewrite(r) }
    }
}

public extension MarkdownInlineNode {
    func rewrite(_ r: (MarkdownInlineNode) throws -> [MarkdownInlineNode]) rethrows -> [MarkdownInlineNode] {
        var inline = self
        inline.children = try children.rewrite(r)
        return try r(inline)
    }
}
