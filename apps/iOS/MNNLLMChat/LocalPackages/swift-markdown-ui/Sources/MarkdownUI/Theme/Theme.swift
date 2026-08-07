import SwiftUI

/// A type that applies a custom appearance to blocks and text inlines in a Markdown view.
///
/// A theme combines the different text and block styles you can customize in a ``Markdown`` view.
///
/// You can set the current Markdown theme in a view hierarchy by using the `markdownTheme(_:)` modifier.
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
/// To create a theme, start by instantiating an empty `Theme` and chain together the different text and
/// block styles in a single expression.
///
/// ```swift
/// let myTheme = Theme()
///   .code {
///     FontFamilyVariant(.monospaced)
///     FontSize(.em(0.85))
///   }
///   .link {
///     ForegroundColor(.purple)
///   }
///   // More text styles...
///   .paragraph { configuration in
///     configuration.label
///       .relativeLineSpacing(.em(0.25))
///       .markdownMargin(top: 0, bottom: 16)
///   }
///   .listItem { configuration in
///     configuration.label
///       .markdownMargin(top: .em(0.25))
///   }
///   // More block styles...
/// ```
public struct Theme {
  /// The default text style.
  public var text: TextStyle = EmptyTextStyle()

  /// The inline code style.
  public var code: TextStyle = FontFamilyVariant(.monospaced)

  /// The emphasis style.
  public var emphasis: TextStyle = FontStyle(.italic)

  /// The strong style.
  public var strong: TextStyle = FontWeight(.semibold)

  /// The strikethrough style.
  public var strikethrough: TextStyle = StrikethroughStyle(.single)

  /// The link style.
  public var link: TextStyle = EmptyTextStyle()

  var headings = Array(
    repeating: BlockStyle<BlockConfiguration> { $0.label },
    count: 6
  )

  /// The level 1 heading style.
  public var heading1: BlockStyle<BlockConfiguration> {
    get { self.headings[0] }
    set { self.headings[0] = newValue }
  }

  /// The level 2 heading style.
  public var heading2: BlockStyle<BlockConfiguration> {
    get { self.headings[1] }
    set { self.headings[1] = newValue }
  }

  /// The level 3 heading style.
  public var heading3: BlockStyle<BlockConfiguration> {
    get { self.headings[2] }
    set { self.headings[2] = newValue }
  }

  /// The level 4 heading style.
  public var heading4: BlockStyle<BlockConfiguration> {
    get { self.headings[3] }
    set { self.headings[3] = newValue }
  }

  /// The level 5 heading style.
  public var heading5: BlockStyle<BlockConfiguration> {
    get { self.headings[4] }
    set { self.headings[4] = newValue }
  }

  /// The level 6 heading style.
  public var heading6: BlockStyle<BlockConfiguration> {
    get { self.headings[5] }
    set { self.headings[5] = newValue }
  }

  /// The paragraph style.
  public var paragraph = BlockStyle<BlockConfiguration> { $0.label }

  /// The blockquote style.
  public var blockquote = BlockStyle<BlockConfiguration> { $0.label }

  /// The code block style.
  public var codeBlock = BlockStyle<CodeBlockConfiguration> { $0.label }

  /// The image style.
  public var image = BlockStyle<BlockConfiguration> { $0.label }

  /// The list style.
  public var list = BlockStyle<BlockConfiguration> { $0.label }

  /// The list item style.
  public var listItem = BlockStyle<BlockConfiguration> { $0.label }

  /// The task list marker style.
  public var taskListMarker = BlockStyle.checkmarkSquare

  /// The bulleted list marker style.
  public var bulletedListMarker = BlockStyle.discCircleSquare

  /// The numbered list marker style.
  public var numberedListMarker = BlockStyle.decimal

  /// The table style.
  public var table = BlockStyle<BlockConfiguration> { $0.label }

  /// The table cell style.
  public var tableCell = BlockStyle<TableCellConfiguration> { $0.label }

  /// The thematic break style.
  public var thematicBreak = BlockStyle { Divider() }

  /// Creates a theme with default text styles.
  public init() {}
}

extension Theme {
  /// Adds a default text style to the theme.
  /// - Parameter text: A text style builder that returns the default text style.
  public func text<S: TextStyle>(@TextStyleBuilder text: () -> S) -> Theme {
    var theme = self
    theme.text = text()
    return theme
  }

  /// Adds an inline code style to the theme.
  /// - Parameter code: A text style builder that returns the inline code style.
  public func code<S: TextStyle>(@TextStyleBuilder code: () -> S) -> Theme {
    var theme = self
    theme.code = code()
    return theme
  }

  /// Adds an emphasis style to the theme.
  /// - Parameter emphasis: A text style builder that returns the emphasis style.
  public func emphasis<S: TextStyle>(@TextStyleBuilder emphasis: () -> S) -> Theme {
    var theme = self
    theme.emphasis = emphasis()
    return theme
  }

  /// Adds a strong style to the theme.
  /// - Parameter strong: A text style builder that returns the strong style.
  public func strong<S: TextStyle>(@TextStyleBuilder strong: () -> S) -> Theme {
    var theme = self
    theme.strong = strong()
    return theme
  }

  /// Adds a strikethrough style to the theme.
  /// - Parameter strikethrough: A text style builder that returns the strikethrough style.
  public func strikethrough<S: TextStyle>(@TextStyleBuilder strikethrough: () -> S) -> Theme {
    var theme = self
    theme.strikethrough = strikethrough()
    return theme
  }

  /// Adds a link style to the theme.
  /// - Parameter link: A text style builder that returns the link style.
  public func link<S: TextStyle>(@TextStyleBuilder link: () -> S) -> Theme {
    var theme = self
    theme.link = link()
    return theme
  }
}

extension Theme {
  /// Adds a level 1 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 1 heading.
  public func heading1<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading1 = .init(body: body)
    return theme
  }

  /// Adds a level 2 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 2 heading.
  public func heading2<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading2 = .init(body: body)
    return theme
  }

  /// Adds a level 3 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 3 heading.
  public func heading3<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading3 = .init(body: body)
    return theme
  }

  /// Adds a level 4 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 4 heading.
  public func heading4<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading4 = .init(body: body)
    return theme
  }

  /// Adds a level 5 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 5 heading.
  public func heading5<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading5 = .init(body: body)
    return theme
  }

  /// Adds a level 6 heading style to the theme.
  /// - Parameter body: A view builder that returns a customized level 6 heading.
  public func heading6<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.heading6 = .init(body: body)
    return theme
  }

  /// Adds a paragraph style to the theme.
  /// - Parameter body: A view builder that returns a customized paragraph.
  public func paragraph<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.paragraph = .init(body: body)
    return theme
  }

  /// Adds a blockquote style to the theme.
  /// - Parameter body: A view builder that returns a customized blockquote.
  public func blockquote<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.blockquote = .init(body: body)
    return theme
  }

  /// Adds a code block style to the theme.
  /// - Parameter body: A view builder that returns a customized code block.
  public func codeBlock<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: CodeBlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.codeBlock = .init(body: body)
    return theme
  }

  /// Adds an image style to the theme.
  /// - Parameter body: A view builder that returns a customized image.
  public func image<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.image = .init(body: body)
    return theme
  }

  /// Adds a list style to the theme.
  /// - Parameter body: A view builder that returns a customized list.
  public func list<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.list = .init(body: body)
    return theme
  }

  /// Adds a list item style to the theme.
  /// - Parameter body: A view builder that returns a customized list item.
  public func listItem<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.listItem = .init(body: body)
    return theme
  }

  /// Adds a task list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized task list marker.
  public func taskListMarker(_ taskListMarker: BlockStyle<TaskListMarkerConfiguration>) -> Theme {
    var theme = self
    theme.taskListMarker = taskListMarker
    return theme
  }

  /// Adds a task list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized task list marker.
  public func taskListMarker<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: TaskListMarkerConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.taskListMarker = .init(body: body)
    return theme
  }

  /// Adds a bulleted list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized bulleted list marker.
  public func bulletedListMarker(
    _ bulletedListMarker: BlockStyle<ListMarkerConfiguration>
  ) -> Theme {
    var theme = self
    theme.bulletedListMarker = bulletedListMarker
    return theme
  }

  /// Adds a bulleted list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized bulleted list marker.
  public func bulletedListMarker<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: ListMarkerConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.bulletedListMarker = .init(body: body)
    return theme
  }

  /// Adds a numbered list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized numbered list marker.
  public func numberedListMarker(
    _ numberedListMarker: BlockStyle<ListMarkerConfiguration>
  ) -> Theme {
    var theme = self
    theme.numberedListMarker = numberedListMarker
    return theme
  }

  /// Adds a numbered list marker style to the theme.
  /// - Parameter body: A view builder that returns a customized numbered list marker.
  public func numberedListMarker<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: ListMarkerConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.numberedListMarker = .init(body: body)
    return theme
  }

  /// Adds a table style to the theme.
  /// - Parameter body: A view builder that returns a customized table.
  public func table<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.table = .init(body: body)
    return theme
  }

  /// Adds a table cell style to the theme.
  /// - Parameter body: A view builder that returns a customized table cell.
  public func tableCell<Body: View>(
    @ViewBuilder body: @escaping (_ configuration: TableCellConfiguration) -> Body
  ) -> Theme {
    var theme = self
    theme.tableCell = .init(body: body)
    return theme
  }

  /// Adds a thematic break style to the theme.
  /// - Parameter body: A view builder that returns a customized thematic break.
  public func thematicBreak<Body: View>(@ViewBuilder body: @escaping () -> Body) -> Theme {
    var theme = self
    theme.thematicBreak = .init(body: body)
    return theme
  }
}

extension Theme {
  /// The text background color of the theme extracted from the ``Theme/text`` style.
  public var textBackgroundColor: Color? {
    var attributes = AttributeContainer()
    self.text._collectAttributes(in: &attributes)
    return attributes.backgroundColor
  }
}
