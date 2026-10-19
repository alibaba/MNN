import SwiftUI

struct HeadingView: View {
  @Environment(\.theme.headings) private var headings

  private let level: Int
  private let content: [InlineNode]

  init(level: Int, content: [InlineNode]) {
    self.level = level
    self.content = content
  }

  var body: some View {
    self.headings[self.level - 1].makeBody(
      configuration: .init(
        label: .init(InlineText(self.content)),
        content: .init(block: .heading(level: self.level, content: self.content))
      )
    )
    .id(content.renderPlainText().kebabCased())
  }
}
