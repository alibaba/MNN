import MarkdownUI
import SwiftUI

struct ThemeOption: Hashable {
  let name: String
  let theme: Theme

  static func == (lhs: ThemeOption, rhs: ThemeOption) -> Bool {
    lhs.name == rhs.name
  }

  func hash(into hasher: inout Hasher) {
    hasher.combine(self.name)
  }

  static let basic = ThemeOption(name: "Basic", theme: .basic)
  static let docC = ThemeOption(name: "DocC", theme: .docC)
  static let gitHub = ThemeOption(name: "GitHub", theme: .gitHub)
}

struct DemoView<Content: View>: View {
  private let themeOptions: [ThemeOption]
  private let about: MarkdownContent?
  private let content: Content

  @State private var themeOption = ThemeOption(name: "Basic", theme: .basic)

  init(
    themeOptions: [ThemeOption] = [.gitHub, .docC, .basic],
    @ViewBuilder content: () -> Content
  ) {
    self.themeOptions = themeOptions
    self.about = nil
    self.content = content()
  }

  init(
    themeOptions: [ThemeOption] = [.gitHub, .docC, .basic],
    @MarkdownContentBuilder about: () -> MarkdownContent,
    @ViewBuilder content: () -> Content
  ) {
    self.themeOptions = themeOptions
    self.about = about()
    self.content = content()
  }

  var body: some View {
    Form {
      if let about {
        Section {
          DisclosureGroup("About this demo") {
            Markdown {
              about
            }
          }
        }
      }

      if !self.themeOptions.isEmpty {
        Section {
          Picker("Theme", selection: $themeOption) {
            ForEach(self.themeOptions, id: \.self) { option in
              Text(option.name).tag(option)
            }
          }
        }
      }

      self.content
        .textSelection(.enabled)
        .markdownTheme(self.themeOption.theme)
        // Some themes may have a custom background color that we need to set as
        // the row's background color.
        .listRowBackground(self.themeOption.theme.textBackgroundColor)
        // By resetting the state when the theme changes, we avoid mixing the
        // the previous theme block spacing preferences with the new theme ones,
        // which can only happen in this particular use case.
        .id(self.themeOption.name)
    }
    .onAppear {
      self.themeOption = self.themeOptions.first ?? .basic
    }
  }
}

struct DemoView_Previews: PreviewProvider {
  static var previews: some View {
    DemoView {
      "Add some text **describing** what this demo is about."
    } content: {
      Markdown {
        Heading(.level2) {
          "Title"
        }
        "Show an awesome **MarkdownUI** feature!"
      }
    }

  }
}
