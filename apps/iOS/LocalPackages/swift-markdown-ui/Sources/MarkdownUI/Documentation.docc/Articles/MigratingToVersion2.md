# Migrating to MarkdownUI 2

Learn how to migrate existing apps using MarkdownUI 1.x to the latest version of the library.

## Overview

MarkdownUI 2 has been rewritten from scratch and brings a ton of new features and improvements
like:

- [GitHub Flavored Markdown](https://github.github.com/gfm/) (tables, task lists,
  strikethrough text, and autolinks)
- Native SwiftUI rendering
- Customization support via themes, text styles, and block styles.

These new features come with the cost of a few breaking changes that this guide will help you to
address.

## Minimum requirements

You can use MarkdownUI 2 on the following platforms:

- macOS 12.0+
- iOS 15.0+
- tvOS 15.0+
- watchOS 8.0+

Some features, like displaying tables or multi-image paragraphs, require macOS 13.0+, iOS 16.0+,
tvOS 16.0+, and watchOS 9.0+.

## Creating Markdown content

MarkdownUI 2 introduces a new domain-specific language to create Markdown content and no longer
depends on [gonzalezreal/SwiftCommonMark](https://github.com/gonzalezreal/SwiftCommonMark).

One significant difference when using MarkdownUI 2 is that ``MarkdownContent`` replaces `Document`
by providing similar functionality.

Another thing to be aware of is the different naming of some of the types you use to compose
Markdown content:

- Use ``Blockquote`` instead of `BlockQuote`.
- Use ``NumberedList`` instead of `OrderedList`.
- Use ``BulletedList`` instead of `BulletList`.
- Use ``InlineImage`` instead of `Image`.
- Use ``InlineLink`` instead of `Link`.
- Use ``Code`` instead of `InlineCode`.

## Loading asset images

MarkdownUI 2 introduces the ``ImageProvider`` protocol and its conforming types
``DefaultImageProvider`` and ``AssetImageProvider``. These types and the new
`markdownImageProvider(_:)` modifier replace the `MarkdownImageHandler` type and
the `setImageHandler(_:forURLScheme:)` modifier.

The following example shows how to configure the asset image provider to load images from the
main bundle.

```swift
Markdown {
  "![A dog](dog)"
  "― Photo by André Spieker"
}
.markdownImageProvider(.asset)
```

## Customizing link behavior

The `onOpenMarkdownLink(perform:)` modifier in MarkdownUI 1.x was provided to enable link behavior
customization in macOS 11.0, iOS 14.0, and tvOS 14.0. This modifier is no longer available in
MarkdownUI 2 since it does not support those platforms. However, you can customize the link
behavior by setting the `openURL` environment value with a custom `OpenURLAction`.

## Styling Markdown

MarkdownUI 1.x offered a few options to customize the content appearance. In contrast, MarkdownUI 2
brings the new ``Theme``, ``TextStyle``, and ``BlockStyle`` types that let you apply a custom
appearance to blocks and text inlines in a Markdown view.

Consequently, the `MarkdownStyle` type, all of its subtypes, and the `markdownStyle(_:)` modifier
are no longer available in MarkdownUI 2.
