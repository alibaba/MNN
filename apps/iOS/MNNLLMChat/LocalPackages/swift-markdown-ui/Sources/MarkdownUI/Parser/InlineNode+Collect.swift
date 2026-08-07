import Foundation

extension Sequence where Element == InlineNode {
  func collect<Result>(_ c: (InlineNode) throws -> [Result]) rethrows -> [Result] {
    try self.flatMap { try $0.collect(c) }
  }
}

extension InlineNode {
  func collect<Result>(_ c: (InlineNode) throws -> [Result]) rethrows -> [Result] {
    try self.children.collect(c) + c(self)
  }
}
