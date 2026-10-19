import SwiftUI

// MARK: - Deprecated after 2.1.0:

extension DefaultImageProvider {
  @available(*, deprecated, message: "Use the 'default' static property")
  public init(urlSession: URLSession = .shared) {
    self.init()
  }
}

extension DefaultInlineImageProvider {
  @available(*, deprecated, message: "Use the 'default' static property")
  public init(urlSession: URLSession = .shared) {
    self.init()
  }
}

// MARK: - Deprecated after 2.0.2:

extension BlockStyle where Configuration == BlockConfiguration {
  @available(
    *,
    deprecated,
    message: "Use the initializer that takes a closure receiving a 'Configuration' value."
  )
  public init<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) {
    self.init { configuration in
      body(configuration.label)
    }
  }

  @available(
    *,
    deprecated,
    message: "Use the initializer that takes a closure receiving a 'Configuration' value."
  )
  public init() {
    self.init { $0 }
  }
}

extension View {
  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a generic 'Configuration'
      value.
      """
  )
  public func markdownBlockStyle<Body: View>(
    _ keyPath: WritableKeyPath<Theme, BlockStyle<BlockConfiguration>>,
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> some View {
    self.environment((\EnvironmentValues.theme).appending(path: keyPath), .init(body: body))
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a generic 'Configuration'
      value.
      """
  )
  public func markdownBlockStyle<Body: View>(
    _ keyPath: WritableKeyPath<Theme, BlockStyle<CodeBlockConfiguration>>,
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> some View {
    self.environment(
      (\EnvironmentValues.theme).appending(path: keyPath),
      .init { configuration in
        body(.init(configuration.label))
      }
    )
  }
}

extension Theme {
  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading1<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading1 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading2<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading2 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading3<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading3 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading4<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading4 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading5<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading5 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func heading6<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.heading6 = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func paragraph<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.paragraph = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func blockquote<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.blockquote = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'CodeBlockConfiguration'
      value.
      """
  )
  public func codeBlock<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.codeBlock = .init { configuration in
      body(.init(configuration.label))
    }
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func image<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.image = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func list<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.list = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func listItem<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.listItem = .init(body: body)
    return theme
  }

  @available(
    *,
    deprecated,
    message: """
      Use the version of this function that takes a closure receiving a 'BlockConfiguration'
      value.
      """
  )
  public func table<Body: View>(
    @ViewBuilder body: @escaping (_ label: BlockConfiguration.Label) -> Body
  ) -> Theme {
    var theme = self
    theme.table = .init(body: body)
    return theme
  }
}

// MARK: - Unavailable after 1.1.1:

extension Heading {
  @available(*, unavailable, message: "Use 'init(_ level:content:)'")
  public init(level: Int, @InlineContentBuilder content: () -> InlineContent) {
    fatalError("Unimplemented")
  }
}

@available(*, unavailable, renamed: "Blockquote")
public typealias BlockQuote = Blockquote

@available(*, unavailable, renamed: "NumberedList")
public typealias OrderedList = NumberedList

@available(*, unavailable, renamed: "BulletedList")
public typealias BulletList = BulletedList

@available(*, unavailable, renamed: "Code")
public typealias InlineCode = Code

@available(
  *,
  unavailable,
  message: """
    "MarkdownImageHandler" has been superseded by the "ImageProvider" protocol and its conforming
    types "DefaultImageProvider" and "AssetImageProvider".
    """
)
public struct MarkdownImageHandler {
  public static var networkImage: Self {
    fatalError("Unimplemented")
  }

  public static func assetImage(
    name: @escaping (URL) -> String = \.lastPathComponent,
    in bundle: Bundle? = nil
  ) -> Self {
    fatalError("Unimplemented")
  }
}

extension Markdown {
  @available(
    *,
    unavailable,
    message: """
      "MarkdownImageHandler" has been superseded by the "ImageProvider" protocol and its conforming
      types "DefaultImageProvider" and "AssetImageProvider".
      """
  )
  public func setImageHandler(
    _ imageHandler: MarkdownImageHandler,
    forURLScheme urlScheme: String
  ) -> Markdown {
    fatalError("Unimplemented")
  }
}

extension View {
  @available(
    *,
    unavailable,
    message: "You can create a custom link action by overriding the \"openURL\" environment value."
  )
  public func onOpenMarkdownLink(perform action: ((URL) -> Void)? = nil) -> some View {
    self
  }
}

@available(
  *,
  unavailable,
  message: """
    "MarkdownStyle" and its subtypes have been superseded by the "Theme", "TextStyle", and
    "BlockStyle" types.
    """
)
public struct MarkdownStyle: Hashable {
  public struct Font: Hashable {
    public static var largeTitle: Self { fatalError("Unimplemented") }
    public static var title: Self { fatalError("Unimplemented") }
    public static var title2: Self { fatalError("Unimplemented") }
    public static var title3: Self { fatalError("Unimplemented") }
    public static var headline: Self { fatalError("Unimplemented") }
    public static var subheadline: Self { fatalError("Unimplemented") }
    public static var body: Self { fatalError("Unimplemented") }
    public static var callout: Self { fatalError("Unimplemented") }
    public static var footnote: Self { fatalError("Unimplemented") }
    public static var caption: Self { fatalError("Unimplemented") }
    public static var caption2: Self { fatalError("Unimplemented") }

    public static func system(
      size: CGFloat,
      weight: SwiftUI.Font.Weight = .regular,
      design: SwiftUI.Font.Design = .default
    ) -> Self {
      fatalError("Unimplemented")
    }

    public static func system(
      _ style: SwiftUI.Font.TextStyle,
      design: SwiftUI.Font.Design = .default
    ) -> Self {
      fatalError("Unimplemented")
    }

    public static func custom(_ name: String, size: CGFloat) -> Self {
      fatalError("Unimplemented")
    }

    public func bold() -> Self {
      fatalError("Unimplemented")
    }

    public func italic() -> Self {
      fatalError("Unimplemented")
    }

    public func monospacedDigit() -> Self {
      fatalError("Unimplemented")
    }

    public func monospaced() -> Self {
      fatalError("Unimplemented")
    }

    public func scale(_ scale: CGFloat) -> Self {
      fatalError("Unimplemented")
    }
  }

  public struct HeadingScales: Hashable {
    public init(
      h1: CGFloat,
      h2: CGFloat,
      h3: CGFloat,
      h4: CGFloat,
      h5: CGFloat,
      h6: CGFloat
    ) {
      fatalError("Unimplemented")
    }

    public subscript(index: Int) -> CGFloat {
      fatalError("Unimplemented")
    }

    public static var `default`: Self {
      fatalError("Unimplemented")
    }
  }

  public struct Measurements: Hashable {
    public var codeFontScale: CGFloat
    public var headIndentStep: CGFloat
    public var tailIndentStep: CGFloat
    public var paragraphSpacing: CGFloat
    public var listMarkerSpacing: CGFloat
    public var headingScales: HeadingScales
    public var headingSpacing: CGFloat

    public init(
      codeFontScale: CGFloat = 0.94,
      headIndentStep: CGFloat = 1.97,
      tailIndentStep: CGFloat = -1,
      paragraphSpacing: CGFloat = 1,
      listMarkerSpacing: CGFloat = 0.47,
      headingScales: HeadingScales = .default,
      headingSpacing: CGFloat = 0.67
    ) {
      fatalError("Unimplemented")
    }
  }

  public var font: MarkdownStyle.Font
  public var foregroundColor: SwiftUI.Color
  public var measurements: Measurements

  public init(
    font: MarkdownStyle.Font = .body,
    foregroundColor: SwiftUI.Color = .primary,
    measurements: MarkdownStyle.Measurements = .init()
  ) {
    fatalError("Unimplemented")
  }
}

extension View {
  @available(
    *,
    unavailable,
    message: """
      "MarkdownStyle" and its subtypes have been superseded by the "Theme", "TextStyle", and
      "BlockStyle" types.
      """
  )
  public func markdownStyle(_ markdownStyle: MarkdownStyle) -> some View {
    self
  }
}
