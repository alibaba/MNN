import MarkdownUI
import Splash
import SwiftUI

struct CodeSyntaxHighlightView: View {
  @Environment(\.colorScheme) private var colorScheme

  private let content = #"""
    This screen demonstrates how you can integrate a 3rd party library
    to render syntax-highlighted code blocks.

    First, we create a type that conforms to `CodeSyntaxHighlighter`,
    using [John Sundell's Splash](https://github.com/JohnSundell/Splash)
    to highlight code blocks.

    ```swift
    import MarkdownUI
    import Splash
    import SwiftUI

    struct SplashCodeSyntaxHighlighter: CodeSyntaxHighlighter {
      private let syntaxHighlighter: SyntaxHighlighter<TextOutputFormat>

      init(theme: Splash.Theme) {
        self.syntaxHighlighter = SyntaxHighlighter(format: TextOutputFormat(theme: theme))
      }

      func highlightCode(_ content: String, language: String?) -> Text {
        guard language != nil else {
          return Text(content)
        }

        return self.syntaxHighlighter.highlight(content)
      }
    }

    extension CodeSyntaxHighlighter where Self == SplashCodeSyntaxHighlighter {
      static func splash(theme: Splash.Theme) -> Self {
        SplashCodeSyntaxHighlighter(theme: theme)
      }
    }
    ```

    Then we configure the `Markdown` view to use the `SplashCodeSyntaxHighlighter`
    that we just created.

    ```swift
    var body: some View {
      Markdown(self.content)
        .markdownCodeSyntaxHighlighter(.splash(theme: .sunset(withFont: .init(size: 16))))
    }
    ```

    More languages to render:

    ```
    A plain code block without the specifying a language name.
    ```

    ```cpp
    #include <iostream>
    #include <vector>

    int main() {
        std::vector<std::string> fruits = {"apple", "banana", "orange"};
        for (const std::string& fruit : fruits) {
            std::cout << "I love " << fruit << "s!" << std::endl;
        }
        return 0;
    }
    ```

    ```typescript
    interface Person {
      name: string;
      age: number;
    }

    const person = Person();
    ```

    ```ruby
    fruits = ["apple", "banana", "orange"]
    fruits.each do |fruit|
      puts "I love #{fruit}s!"
    end
    ```

    """#

  var body: some View {
    DemoView {
      Markdown(self.content)
        .markdownBlockStyle(\.codeBlock) {
          codeBlock($0)
        }
        .markdownCodeSyntaxHighlighter(.splash(theme: self.theme))
    }
  }

  @ViewBuilder
  private func codeBlock(_ configuration: CodeBlockConfiguration) -> some View {
    VStack(spacing: 0) {
      HStack {
        Text(configuration.language ?? "plain text")
          .font(.system(.caption, design: .monospaced))
          .fontWeight(.semibold)
          .foregroundColor(Color(theme.plainTextColor))
        Spacer()

        Image(systemName: "clipboard")
          .onTapGesture {
            copyToClipboard(configuration.content)
          }
      }
      .padding(.horizontal)
      .padding(.vertical, 8)
      .background {
        Color(theme.backgroundColor)
      }

      Divider()

      ScrollView(.horizontal) {
        configuration.label
          .relativeLineSpacing(.em(0.25))
          .markdownTextStyle {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.85))
          }
          .padding()
      }
    }
    .background(Color(.secondarySystemBackground))
    .clipShape(RoundedRectangle(cornerRadius: 8))
    .markdownMargin(top: .zero, bottom: .em(0.8))
  }

  private var theme: Splash.Theme {
    // NOTE: We are ignoring the Splash theme font
    switch self.colorScheme {
    case .dark:
      return .wwdc17(withFont: .init(size: 16))
    default:
      return .sunset(withFont: .init(size: 16))
    }
  }

  private func copyToClipboard(_ string: String) {
    #if os(macOS)
      if let pasteboard = NSPasteboard.general {
        pasteboard.clearContents()
        pasteboard.setString(string, forType: .string)
      }
    #elseif os(iOS)
      UIPasteboard.general.string = string
    #endif
  }
}

struct CodeSyntaxHighlightView_Previews: PreviewProvider {
  static var previews: some View {
    CodeSyntaxHighlightView()
  }
}
