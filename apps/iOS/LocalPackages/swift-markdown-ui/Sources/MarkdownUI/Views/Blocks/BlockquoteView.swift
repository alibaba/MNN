import SwiftUI

struct BlockquoteView: View {
  @Environment(\.theme.blockquote) private var blockquote

  private let children: [BlockNode]

  init(children: [BlockNode]) {
    self.children = children
  }

  var body: some View {
    self.blockquote.makeBody(
      configuration: .init(
        label: .init(BlockSequence(self.children)),
        content: .init(block: .blockquote(children: self.children))
      )
    )
  }
}
