import Foundation

enum InlineNode: Hashable, Sendable {
  case text(String)
  case softBreak
  case lineBreak
  case code(String)
  case html(String)
  case emphasis(children: [InlineNode])
  case strong(children: [InlineNode])
  case strikethrough(children: [InlineNode])
  case link(destination: String, children: [InlineNode])
  case image(source: String, children: [InlineNode])
}

extension InlineNode {
  var children: [InlineNode] {
    get {
      switch self {
      case .emphasis(let children):
        return children
      case .strong(let children):
        return children
      case .strikethrough(let children):
        return children
      case .link(_, let children):
        return children
      case .image(_, let children):
        return children
      default:
        return []
      }
    }

    set {
      switch self {
      case .emphasis:
        self = .emphasis(children: newValue)
      case .strong:
        self = .strong(children: newValue)
      case .strikethrough:
        self = .strikethrough(children: newValue)
      case .link(let destination, _):
        self = .link(destination: destination, children: newValue)
      case .image(let source, _):
        self = .image(source: source, children: newValue)
      default:
        break
      }
    }
  }
}
