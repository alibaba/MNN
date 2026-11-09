import SwiftUI

extension View {
  /// Sets the code syntax highlighter for the Markdown code blocks in a view hierarchy.
  /// - Parameter codeSyntaxHighlighter: The code syntax highlighter to set. Use
  ///                                    ``CodeSyntaxHighlighter/plainText`` or a custom syntax
  ///                                    highlighter that you define by creating a type that conforms to the
  ///                                    ``CodeSyntaxHighlighter`` protocol.
  /// - Returns: A view that uses the specified code syntax highlighter for itself and its
  ///            child views.
  public func markdownCodeSyntaxHighlighter(
    _ codeSyntaxHighlighter: CodeSyntaxHighlighter
  ) -> some View {
    self.environment(\.codeSyntaxHighlighter, codeSyntaxHighlighter)
  }
}

extension EnvironmentValues {
  var codeSyntaxHighlighter: CodeSyntaxHighlighter {
    get { self[CodeSyntaxHighlighterKey.self] }
    set { self[CodeSyntaxHighlighterKey.self] = newValue }
  }
}

private struct CodeSyntaxHighlighterKey: EnvironmentKey {
  static let defaultValue: CodeSyntaxHighlighter = .plainText
}
