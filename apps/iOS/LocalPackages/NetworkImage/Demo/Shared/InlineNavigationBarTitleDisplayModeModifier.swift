import SwiftUI

extension View {
  func inlineNavigationBarTitleDisplayMode() -> some View {
    self.modifier(InlineNavigationBarTitleDisplayModeModifier())
  }
}

struct InlineNavigationBarTitleDisplayModeModifier: ViewModifier {
  func body(content: Content) -> some View {
    #if os(iOS) || os(watchOS)
      content.navigationBarTitleDisplayMode(.inline)
    #else
      content
    #endif
  }
}
