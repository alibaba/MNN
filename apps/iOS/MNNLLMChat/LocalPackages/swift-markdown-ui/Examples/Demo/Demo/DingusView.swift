import MarkdownUI
import SwiftUI

struct DingusView: View {
  @State private var markdown = """
    ## Try GitHub Flavored Markdown

    You can try **GitHub Flavored Markdown** here.  This dingus is powered
    by [MarkdownUI](https://github.com/gonzalezreal/MarkdownUI), a native
    Markdown renderer for SwiftUI.

    1. item one
    1. item two
       - sublist
       - sublist
    """

  var body: some View {
    DemoView {
      Section("Editor") {
        TextEditor(text: $markdown)
          .font(.system(.callout, design: .monospaced))
      }

      Section("Preview") {
        Markdown(self.markdown)
      }
    }
  }
}

struct DingusView_Previews: PreviewProvider {
  static var previews: some View {
    DingusView()
  }
}
