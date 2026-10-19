import SwiftUI

/// The appearance of a text inline in a Markdown view.
///
/// The styles of the different text inline types are brought together in a ``Theme``. You can customize the style of a
/// specific inline type by using the `markdownTextStyle(_:textStyle:)` modifier and combining one or more
/// built-in text styles like ``ForegroundColor`` or ``FontWeight``.
///
/// The following example applies a custom text style to each ``Theme/code`` inline in a ``Markdown`` view.
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
/// You can also override the default text style inside the body of any block style by using the `markdownTextStyle(textStyle:)`
/// modifier. For example, you can define a ``Theme/blockquote`` style that uses a semibold lowercase small-caps text style:
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
public protocol TextStyle {
  func _collectAttributes(in attributes: inout AttributeContainer)
}
