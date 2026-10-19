import Foundation

extension Sequence where Element == BlockNode {
  func rewrite(_ r: (BlockNode) throws -> [BlockNode]) rethrows -> [BlockNode] {
    try self.flatMap { try $0.rewrite(r) }
  }

  func rewrite(_ r: (InlineNode) throws -> [InlineNode]) rethrows -> [BlockNode] {
    try self.flatMap { try $0.rewrite(r) }
  }
}

extension BlockNode {
  func rewrite(_ r: (BlockNode) throws -> [BlockNode]) rethrows -> [BlockNode] {
    switch self {
    case .blockquote(let children):
      return try r(.blockquote(children: children.rewrite(r)))
    case .bulletedList(let isTight, let items):
      return try r(
        .bulletedList(
          isTight: isTight,
          items: try items.map {
            RawListItem(children: try $0.children.rewrite(r))
          }
        )
      )
    case .numberedList(let isTight, let start, let items):
      return try r(
        .numberedList(
          isTight: isTight,
          start: start,
          items: try items.map {
            RawListItem(children: try $0.children.rewrite(r))
          }
        )
      )
    case .taskList(let isTight, let items):
      return try r(
        .taskList(
          isTight: isTight,
          items: try items.map {
            RawTaskListItem(isCompleted: $0.isCompleted, children: try $0.children.rewrite(r))
          }
        )
      )
    default:
      return try r(self)
    }
  }

  func rewrite(_ r: (InlineNode) throws -> [InlineNode]) rethrows -> [BlockNode] {
    switch self {
    case .blockquote(let children):
      return [.blockquote(children: try children.rewrite(r))]
    case .bulletedList(let isTight, let items):
      return [
        .bulletedList(
          isTight: isTight,
          items: try items.map {
            RawListItem(children: try $0.children.rewrite(r))
          }
        )
      ]
    case .numberedList(let isTight, let start, let items):
      return [
        .numberedList(
          isTight: isTight,
          start: start,
          items: try items.map {
            RawListItem(children: try $0.children.rewrite(r))
          }
        )
      ]
    case .taskList(let isTight, let items):
      return [
        .taskList(
          isTight: isTight,
          items: try items.map {
            RawTaskListItem(isCompleted: $0.isCompleted, children: try $0.children.rewrite(r))
          }
        )
      ]
    case .paragraph(let content):
      return [.paragraph(content: try content.rewrite(r))]
    case .heading(let level, let content):
      return [.heading(level: level, content: try content.rewrite(r))]
    case .table(let columnAlignments, let rows):
      return [
        .table(
          columnAlignments: columnAlignments,
          rows: try rows.map {
            RawTableRow(
              cells: try $0.cells.map {
                RawTableCell(content: try $0.content.rewrite(r))
              }
            )
          }
        )
      ]
    default:
      return [self]
    }
  }
}
