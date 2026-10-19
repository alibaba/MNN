/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

#if os(macOS)

import Cocoa
import Splash

guard let options = CommandLine.makeOptions() else {
    print("""
    ⚠️  Two arguments are required:
    - The code to generate an image for
    - The path to write the generated image to

    Optionally, the following arguments can be passed:
    -p The amount of padding (in pixels) to apply around the code
    -f A path to a font to use when rendering
    -s The size of text to use when rendering
    """)
    exit(1)
}

let theme = Theme.sundellsColors(withFont: options.font)
let outputFormat = AttributedStringOutputFormat(theme: theme)
let highlighter = SyntaxHighlighter(format: outputFormat)
let string = highlighter.highlight(options.code)
let stringSize = string.size()

let contextRect = CGRect(
    x: 0,
    y: 0,
    width: stringSize.width + options.padding * 2,
    height: stringSize.height + options.padding * 2
)

let context = NSGraphicsContext(size: contextRect.size)
NSGraphicsContext.current = context

context.fill(with: theme.backgroundColor, in: contextRect)

string.draw(in: CGRect(
    x: options.padding,
    y: options.padding,
    width: stringSize.width,
    height: stringSize.height
))

let image = context.cgContext.makeImage()!
image.write(to: options.outputURL)

#else
print("😞 SplashImageGen currently only supports macOS")
#endif
