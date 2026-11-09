import MarkdownUI
import SwiftUI

struct QuotesView: View {
  let content = """
    You can quote text with a `>`.

    > Outside of a dog, a book is man's best friend. Inside of a
    > dog it's too dark to read.

    – Groucho Marx
    """

  var body: some View {
    DemoView {
      Markdown(self.content)

      Section("Customization Example") {
        Markdown(self.content)
      }
      .markdownBlockStyle(\.blockquote) { configuration in
        configuration.label
          .padding()
          .markdownTextStyle {
            FontCapsVariant(.lowercaseSmallCaps)
            FontWeight(.semibold)
            BackgroundColor(nil)
          }
          .overlay(alignment: .leading) {
            Rectangle()
              .fill(Color.teal)
              .frame(width: 4)
          }
          .background(Color.teal.opacity(0.5))
      }
    }
  }
}

struct BlockquotesView_Previews: PreviewProvider {
  static var previews: some View {
    QuotesView()
  }
}
