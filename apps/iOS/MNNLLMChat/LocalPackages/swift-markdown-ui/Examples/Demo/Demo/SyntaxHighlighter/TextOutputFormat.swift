import Splash
import SwiftUI

struct TextOutputFormat: OutputFormat {
  private let theme: Theme

  init(theme: Theme) {
    self.theme = theme
  }

  func makeBuilder() -> Builder {
    Builder(theme: self.theme)
  }
}

extension TextOutputFormat {
  struct Builder: OutputBuilder {
    private let theme: Theme
    private var accumulatedText: [Text]

    fileprivate init(theme: Theme) {
      self.theme = theme
      self.accumulatedText = []
    }

    mutating func addToken(_ token: String, ofType type: TokenType) {
      let color = self.theme.tokenColors[type] ?? self.theme.plainTextColor
      self.accumulatedText.append(Text(token).foregroundColor(.init(uiColor: color)))
    }

    mutating func addPlainText(_ text: String) {
      self.accumulatedText.append(
        Text(text).foregroundColor(.init(uiColor: self.theme.plainTextColor))
      )
    }

    mutating func addWhitespace(_ whitespace: String) {
      self.accumulatedText.append(Text(whitespace))
    }

    func build() -> Text {
      self.accumulatedText.reduce(Text(""), +)
    }
  }
}
