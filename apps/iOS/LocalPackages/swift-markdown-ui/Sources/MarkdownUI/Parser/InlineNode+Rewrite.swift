import Foundation

extension Sequence where Element == InlineNode {
  func rewrite(_ r: (InlineNode) throws -> [InlineNode]) rethrows -> [InlineNode] {
    try self.flatMap { try $0.rewrite(r) }
  }
}

extension InlineNode {
  func rewrite(_ r: (InlineNode) throws -> [InlineNode]) rethrows -> [InlineNode] {
    var inline = self
    inline.children = try self.children.rewrite(r)
    return try r(inline)
  }
}
