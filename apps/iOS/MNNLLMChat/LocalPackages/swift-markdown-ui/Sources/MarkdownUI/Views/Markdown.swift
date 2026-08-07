import SwiftUI

/// A view that displays read-only Markdown content.
///
/// A `Markdown` view displays rich structured text using the Markdown syntax. It is compatible with the
/// [GitHub Flavored Markdown Spec](https://github.github.com/gfm/) and can display
/// images, headings, lists (including task lists), blockquotes, code blocks, tables, and thematic breaks,
/// besides styled text and links.
///
/// The simplest way of creating a `Markdown` view is to pass a Markdown string to the
/// ``Markdown/init(_:baseURL:imageBaseURL:)-63py1`` initializer.
///
/// ```swift
/// let markdownString = """
///   ## Try MarkdownUI
///
///   **MarkdownUI** is a native Markdown renderer for SwiftUI
///   compatible with the
///   [GitHub Flavored Markdown Spec](https://github.github.com/gfm/).
///   """
///
/// var body: some View {
///   Markdown(markdownString)
/// }
/// ```
///
/// ![](MarkdownString)
///
/// A more convenient way to create a `Markdown` view is by using the ``Markdown/init(baseURL:imageBaseURL:content:)``
/// initializer, which takes a Markdown content builder in which you can compose the view content, either by providing Markdown strings
/// or by using an expressive domain-specific language.
///
/// ```swift
/// var body: some View {
///   Markdown {
///     """
///     ## Using a Markdown Content Builder
///     Use Markdown strings or an expressive domain-specific language
///     to build the content.
///     """
///     Heading(.level2) {
///       "Try MarkdownUI"
///     }
///     Paragraph {
///       Strong("MarkdownUI")
///       " is a native Markdown renderer for SwiftUI"
///       " compatible with the "
///       InlineLink(
///         "GitHub Flavored Markdown Spec",
///         destination: URL(string: "https://github.github.com/gfm/")!
///       )
///       "."
///     }
///   }
/// }
/// ```
///
/// ![](MarkdownContentBuilder)
///
/// You can also create a ``MarkdownContent`` value in your model layer and later create a `Markdown` view by passing
/// the content value to the ``Markdown/init(_:baseURL:imageBaseURL:)-42bru`` initializer. The ``MarkdownContent``
/// value pre-parses the Markdown string preventing the view from doing this step.
///
/// ```swift
/// // Somewhere in the model layer
/// let content = MarkdownContent("You can try **CommonMark** [here](https://spec.commonmark.org/dingus/).")
///
/// // Later in the view layer
/// var body: some View {
///   Markdown(self.model.content)
/// }
/// ```
///
/// ### Styling Markdown
///
/// Markdown views use a basic default theme to display the contents. For more information, read about the ``Theme/basic`` theme.
///
/// ```swift
/// Markdown {
///   """
///   You can quote text with a `>`.
///
///   > Outside of a dog, a book is man's best friend. Inside of a
///   > dog it's too dark to read.
///
///   – Groucho Marx
///   """
/// }
/// ```
///
/// ![](BlockquoteContent)
///
/// You can customize the appearance of Markdown content by applying different themes using the `markdownTheme(_:)` modifier.
/// For example, you can apply one of the built-in themes, like ``Theme/gitHub``, to either a Markdown view or a view hierarchy that
/// contains Markdown views.
///
/// ```swift
/// Markdown {
///   """
///   You can quote text with a `>`.
///
///   > Outside of a dog, a book is man's best friend. Inside of a
///   > dog it's too dark to read.
///
///   – Groucho Marx
///   """
/// }
/// .markdownTheme(.gitHub)
/// ```
///
/// ![](GitHubBlockquote)
///
/// To override a specific text style from the current theme, use the `markdownTextStyle(_:textStyle:)`
/// modifier.  The following example shows how to override the ``Theme/code`` text style.
///
/// ```swift
/// Markdown {
///   """
///   Use `git status` to list all new or modified files
///   that haven't yet been committed.
///   """
/// }
/// .markdownTextStyle(\.code) {
///   FontFamilyVariant(.monospaced)
///   FontSize(.em(0.85))
///   ForegroundColor(.purple)
///   BackgroundColor(.purple.opacity(0.25))
/// }
/// ```
///
/// ![](CustomInlineCode)
///
/// You can also use the `markdownBlockStyle(_:body:)` modifier to override a specific block style. For example, you can
/// override only the ``Theme/blockquote`` block style, leaving other block styles untouched.
///
/// ```swift
/// Markdown {
///   """
///   You can quote text with a `>`.
///
///   > Outside of a dog, a book is man's best friend. Inside of a
///   > dog it's too dark to read.
///
///   – Groucho Marx
///   """
/// }
/// .markdownBlockStyle(\.blockquote) { configuration in
///   configuration.label
///     .padding()
///     .markdownTextStyle {
///       FontCapsVariant(.lowercaseSmallCaps)
///       FontWeight(.semibold)
///       BackgroundColor(nil)
///     }
///     .overlay(alignment: .leading) {
///       Rectangle()
///         .fill(Color.teal)
///         .frame(width: 4)
///     }
///     .background(Color.teal.opacity(0.5))
/// }
/// ```
///
/// ![](CustomBlockquote)
///
/// Another way to customize the appearance of Markdown content is to create your own theme. See the documentation of the
/// ``Theme`` type for more information on this subject.
///
/// ### Customizing link behavior
///
/// When a user taps or clicks on a Markdown link, the default behavior is to open Safari. However, you can customize this behavior
/// by setting the `openURL` environment value with a custom `OpenURLAction`.
///
/// ```swift
/// Markdown {
///   """
///   ## Try MarkdownUI
///   **MarkdownUI** is a native Markdown renderer for SwiftUI
///   compatible with the
///   [GitHub Flavored Markdown Spec](https://github.github.com/gfm/).
///   """
/// }
/// .environment(
///   \.openURL,
///   OpenURLAction { url in
///     print("Open \(url)")
///     return .handled
///   }
/// )
/// ```
public struct Markdown: View {
  @Environment(\.colorScheme) private var colorScheme
  @Environment(\.theme.text) private var text

  private let content: MarkdownContent
  private let baseURL: URL?
  private let imageBaseURL: URL?

  /// Creates a Markdown view from a Markdown content value.
  /// - Parameters:
  ///   - content: The Markdown content value.
  ///   - baseURL: The base URL to use when resolving Markdown URLs. If this value is `nil`, the initializer will consider all
  ///              URLs absolute. The default is `nil`.
  ///   - imageBaseURL: The base URL to use when resolving Markdown image URLs. If this value is `nil`, the initializer will
  ///                   determine image URLs using the `baseURL` parameter. The default is `nil`.
  public init(_ content: MarkdownContent, baseURL: URL? = nil, imageBaseURL: URL? = nil) {
    self.content = content
    self.baseURL = baseURL
    self.imageBaseURL = imageBaseURL ?? baseURL
  }

  public var body: some View {
    TextStyleAttributesReader { attributes in
      BlockSequence(self.blocks)
        .foregroundColor(attributes.foregroundColor)
        .background(attributes.backgroundColor)
        .modifier(ScaledFontSizeModifier(attributes.fontProperties?.size))
    }
    .textStyle(self.text)
    .environment(\.baseURL, self.baseURL)
    .environment(\.imageBaseURL, self.imageBaseURL)
  }

  private var blocks: [BlockNode] {
    self.content.blocks.filterImagesMatching(colorScheme: self.colorScheme)
  }
}

extension Markdown {
  /// Creates a Markdown view from a Markdown-formatted string.
  /// - Parameters:
  ///   - markdown: The string that contains the Markdown formatting.
  ///   - baseURL: The base URL to use when resolving Markdown URLs. If this value is `nil`, the initializer will consider all
  ///              URLs absolute. The default is `nil`.
  ///   - imageBaseURL: The base URL to use when resolving Markdown image URLs. If this value is `nil`, the initializer will
  ///                   determine image URLs using the `baseURL` parameter. The default is `nil`.
  public init(_ markdown: String, baseURL: URL? = nil, imageBaseURL: URL? = nil) {
    self.init(MarkdownContent(markdown), baseURL: baseURL, imageBaseURL: imageBaseURL)
  }

  /// Creates a Markdown view composed of any number of blocks.
  ///
  /// Using this initializer, you can compose the Markdown view content either by providing Markdown strings or with an expressive
  /// domain-specific language.
  ///
  /// ```swift
  /// var body: some View {
  ///   Markdown {
  ///     """
  ///     ## Using a Markdown Content Builder
  ///
  ///     Use Markdown strings or an expressive domain-specific language
  ///     to build the content.
  ///     """
  ///     Heading(.level2) {
  ///       "Try MarkdownUI"
  ///     }
  ///     Paragraph {
  ///       Strong("MarkdownUI")
  ///       " is a native Markdown renderer for SwiftUI"
  ///       " compatible with the "
  ///       InlineLink(
  ///         "GitHub Flavored Markdown Spec",
  ///         destination: URL(string: "https://github.github.com/gfm/")!
  ///       )
  ///       "."
  ///     }
  ///   }
  /// }
  /// ```
  ///
  /// - Parameters:
  ///   - baseURL: The base URL to use when resolving Markdown URLs. If this value is `nil`, the initializer will consider all
  ///              URLs absolute. The default is `nil`.
  ///   - imageBaseURL: The base URL to use when resolving Markdown image URLs. If this value is `nil`, the initializer will
  ///                   determine image URLs using the `baseURL` parameter. The default is `nil`.
  ///   - content: A Markdown content builder that returns the blocks that form the Markdown content.
  public init(
    baseURL: URL? = nil,
    imageBaseURL: URL? = nil,
    @MarkdownContentBuilder content: () -> MarkdownContent
  ) {
    self.init(content(), baseURL: baseURL, imageBaseURL: imageBaseURL)
  }
}

private struct ScaledFontSizeModifier: ViewModifier {
  @ScaledMetric private var size: CGFloat

  init(_ size: CGFloat?) {
    self._size = ScaledMetric(wrappedValue: size ?? FontProperties.defaultSize, relativeTo: .body)
  }

  func body(content: Content) -> some View {
    content.markdownTextStyle {
      FontSize(self.size)
    }
  }
}
