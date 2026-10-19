# MarkdownView

A powerful pure UIKit framework for rendering Markdown documents with real-time parsing and rendering capabilities. Battle tested in [FlowDown](https://github.com/Lakr233/FlowDown).

## Preview

![Preview](./Resources/Simulator%20Screenshot%20-%20iPad%20mini%20(A17%20Pro)%20-%202025-05-27%20at%2003.03.27.png)

## Features

- 🚀 **Real-time Rendering**: Live Markdown parsing and rendering as you type
- 🖥️ **Specialized for Mobile Display**: Optimized layout that extracts complex elements from lists for better readability
- 🎨 **Syntax Highlighting**: Beautiful code syntax highlighting with Splash
- 📊 **Math Rendering**: LaTeX math formula rendering with SwiftMath
- 📱 **iOS Optimized**: Native UIKit implementation for optimal performance

## Installation

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/Lakr233/MarkdownView", from: "3.4.0"),
]
```

Platform compatibility:
- iOS 13.0+
- Mac Catalyst 13.0+

## Usage

```swift
import MarkdownView
import MarkdownParser

let markdownTextView = MarkdownTextView()
let parser = MarkdownParser()
let result = parser.parse("# Hello World")
let content = MarkdownTextView.PreprocessedContent(parserResult: result, theme: .default)
markdownTextView.setMarkdown(content)
```

## Example

Check out the included example project to see MarkdownView in action:

```bash
cd Example
open Example.xcodeproj
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project includes code adapted from [swift-markdown-ui](https://github.com/gonzalezreal/swift-markdown-ui) by Guillermo Gonzalez, used under the MIT License.

---

Copyright 2025 © Lakr Aream. All rights reserved.