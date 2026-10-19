# ``MarkdownUI``

Display and customize Markdown text in SwiftUI.

## Overview

MarkdownUI is a powerful library for displaying and customizing Markdown text in SwiftUI. It is
compatible with the [GitHub Flavored Markdown Spec](https://github.github.com/gfm/) and can
display images, headings, lists (including task lists), blockquotes, code blocks, tables,
and thematic breaks, besides styled text and links.

MarkdownUI offers comprehensible theming features to customize how it displays Markdown text.
You can use the built-in themes, create your own or override specific text and block styles.

![A MarkdownUI view that displays a heading, paragraph, code block, and table using different themes](MarkdownUI)

## Topics

### Essentials

- <doc:GettingStarted>

### Upgrade guides

- <doc:MigratingToVersion2>

### Displaying Markdown

- ``Markdown``

### Customizing Appearance

- ``Theme``
- ``TextStyle``
- ``ForegroundColor``
- ``BackgroundColor``
- ``FontFamily``
- ``FontSize``
- ``FontStyle``
- ``FontWeight``
- ``FontWidth``
- ``StrikethroughStyle``
- ``UnderlineStyle``
- ``FontFamilyVariant``
- ``FontCapsVariant``
- ``FontDigitVariant``
- ``TextKerning``
- ``TextTracking``
- ``BlockStyle``
- ``BlockConfiguration``
- ``ListMarkerConfiguration``
- ``TaskListMarkerConfiguration``
- ``TableBackgroundStyle``
- ``TableBorderStyle``
- ``TableCellConfiguration``

### Formatting content

- ``MarkdownContent``
- ``Paragraph``
- ``Heading``
- ``Blockquote``
- ``CodeBlock``
- ``BulletedList``
- ``NumberedList``
- ``ListItem``
- ``TaskList``
- ``TaskListItem``
- ``TextTable``
- ``ThematicBreak``
- ``InlineContent``
- ``Code``
- ``Emphasis``
- ``Strong``
- ``Strikethrough``
- ``InlineImage``
- ``InlineLink``

### Extensibility

- ``ImageProvider``
- ``DefaultImageProvider``
- ``AssetImageProvider``
- ``CodeSyntaxHighlighter``
- ``PlainTextCodeSyntaxHighlighter``
