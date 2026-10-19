# SwiftMath

`SwiftMath` provides a full Swift implementation of [iosMath](https://travis-ci.org/kostub/iosMath) 
for displaying beautifully rendered math equations in iOS and MacOS applications. It typesets formulae written 
using LaTeX in a `UILabel` equivalent class. It uses the same typesetting rules as LaTeX and
so the equations are rendered exactly as LaTeX would render them.

Please also check out [SwiftMathDemo](https://github.com/mgriebling/SwiftMathDemo.git) for examples of how to use `SwiftMath`
from SwiftUI.  

`SwiftMath` is similar to [MathJax](https://www.mathjax.org) or
[KaTeX](https://github.com/Khan/KaTeX) for the web but for native iOS or MacOS
applications without having to use a `UIWebView` and Javascript. More
importantly, it is significantly faster than using a `UIWebView`.

`SwiftMath` is a Swift translation of the latest `iosMath` v0.9.5 release but includes bug fixes
and enhancements like a new \lbar (lambda bar) character and cyrillic alphabet support.
The original `iosMath` test suites have also been translated to Swift and run without errors.
Note: Error test conditions are ignored to avoid tagging everything with silly `throw`s.
Please let me know of any bugs or bug fixes that you find. 

`SwiftMath` prepackages everything needed for direct access via the Swift Package Manager.

## Examples
Here are screenshots of some formulae that were rendered with this library:

```LaTeX
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
```

![Quadratic Formula](img/quadratic-light.png#gh-light-mode-only) 
![Quadratic Formula](img/quadratic-dark.png#gh-dark-mode-only) 

```LaTeX
f(x) = \int\limits_{-\infty}^\infty\!\hat f(\xi)\,e^{2 \pi i \xi x}\,\mathrm{d}\xi
```

![Calculus](img/calculus-light.png#gh-light-mode-only) 
![Calculus](img/calculus-dark.png#gh-dark-mode-only) 

```LaTeX
\frac{1}{n}\sum_{i=1}^{n}x_i \geq \sqrt[n]{\prod_{i=1}^{n}x_i}
```

![AM-GM](img/amgm-light.png#gh-light-mode-only) 
![AM-GM](img/amgm-dark.png#gh-dark-mode-only) 

```LaTex
\frac{1}{\left(\sqrt{\phi \sqrt{5}}-\phi\\right) e^{\frac25 \pi}}
= 1+\frac{e^{-2\pi}} {1 +\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}} {1+\frac{e^{-8\pi}} {1+\cdots} } } }
```

![Ramanujan Identity](img/ramanujan-light.png#gh-light-mode-only) 
![Ramanujan Identity](img/ramanujan-dark.png#gh-dark-mode-only) 

More examples are included in [EXAMPLES](EXAMPLES.md)

## Fonts
Here are previews of the included fonts:

![](img/FontsPreview.png#gh-dark-mode-only) 
![](img/FontsPreviewLight.png#gh-light-mode-only) 
 
## Requirements
`SwiftMath` works on iOS 11+ or MacOS 12+. It depends
on the following Apple frameworks:

* Foundation.framework
* CoreGraphics.framework
* QuartzCore.framework
* CoreText.framework

Additionally for iOS it requires:
* UIKit.framework

Additionally for MacOS it requires:
* AppKit.framework

## Installation

### Swift Package

`SwiftMath` is available from [SwiftMath](https://github.com/mgriebling/SwiftMath.git). 
To use it in your code, just add the https://github.com/mgriebling/SwiftMath.git path to
XCode's package manager.

## Usage

The library provides a class `MTMathUILabel` which is a `UIView` that
supports rendering math equations. To display an equation simply create
an `MTMathUILabel` as follows:

```swift

import SwiftMath

let label = MTMathUILabel()
label.latex = "x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}"

```
Adding `MTMathUILabel` as a sub-view of your `UIView` will render the
quadratic formula example shown above.

The following code creates a SwiftUI component called `MathView` encapsulating the MTMathUILabel:

```swift
import SwiftUI
import SwiftMath

struct MathView: UIViewRepresentable {
    var equation: String
    var font: MathFont = .latinModernFont
    var textAlignment: MTTextAlignment = .center
    var fontSize: CGFloat = 30
    var labelMode: MTMathUILabelMode = .text
    var insets: MTEdgeInsets = MTEdgeInsets()
    
    func makeUIView(context: Context) -> MTMathUILabel {
        let view = MTMathUILabel()
        return view
    }
    func updateUIView(_ view: MTMathUILabel, context: Context) {
        view.latex = equation
        view.font = MTFontManager().font(withName: font.rawValue, size: fontSize)
        view.textAlignment = textAlignment
        view.labelMode = labelMode
        view.textColor = MTColor(Color.primary)
        view.contentInsets = insets
    }
}
```

For code that works with SwiftUI running natively under MacOS use the following:

```swift
import SwiftUI
import SwiftMath

struct MathView: NSViewRepresentable {
    var equation: String
    var font: MathFont = .latinModernFont
    var textAlignment: MTTextAlignment = .center
    var fontSize: CGFloat = 30
    var labelMode: MTMathUILabelMode = .text
    var insets: MTEdgeInsets = MTEdgeInsets()
    
    func makeNSView(context: Context) -> MTMathUILabel {
        let view = MTMathUILabel()
        return view
    }
    
    func updateNSView(_ view: MTMathUILabel, context: Context) {
        view.latex = equation
        view.font = MTFontManager().font(withName: font.rawValue, size: fontSize)
        view.textAlignment = textAlignment
        view.labelMode = labelMode
        view.textColor = MTColor(Color.primary)
        view.contentInsets = insets
    }
}
```

### Included Features
This is a list of formula types that the library currently supports:

* Simple algebraic equations
* Fractions and continued fractions
* Exponents and subscripts
* Trigonometric formulae
* Square roots and n-th roots
* Calculus symbos - limits, derivatives, integrals
* Big operators (e.g. product, sum)
* Big delimiters (using \\left and \\right)
* Greek alphabet
* Combinatorics (\\binom, \\choose etc.)
* Geometry symbols (e.g. angle, congruence etc.)
* Ratios, proportions, percentages
* Math spacing
* Overline and underline
* Math accents
* Matrices
* Equation alignment
* Change bold, roman, caligraphic and other font styles (\\bf, \\text, etc.)
* Most commonly used math symbols
* Colors for both text and background

Note: SwiftMath only supports the commands in LaTeX's math mode. There is
also no language support for other than west European langugages and some
Cyrillic characters. There would be two ways to support more languages:

1) Find a math font compatible with `SwiftMath` that contains all the glyphs
for that language.
2) Add support to `SwiftMath` for standard Unicode fonts that contain all
langauge glyphs.

Of these two, the first is much easier.  However, if you want a challenge,
try to tackle the second option.

### Example

The [SwiftMathDemo](https://github.com/mgriebling/SwiftMathDemo) is a SwiftUI version
of the Objective-C demo included in `iosMath` that uses `SwiftMath` as a Swift package dependency.

### Advanced configuration

`MTMathUILabel` supports some advanced configuration options:

##### Math mode

You can change the mode of the `MTMathUILabel` between Display Mode
(equivalent to `$$` or `\[` in LaTeX) and Text Mode (equivalent to `$`
or `\(` in LaTeX). The default style is Display. To switch to Text
simply:

```swift
label.labelMode = .text
```

##### Text Alignment
The default alignment of the equations is left. This can be changed to
center or right as follows:

```swift
label.textAlignment = .center
```

##### Font size
The default font-size is 30pt. You can change it as follows:

```swift
label.fontSize = 25
```
##### Font
The default font is *Latin Modern Math*. This can be changed as:

```swift
label.font = MTFontManager.fontmanager.termesFont(withSize:20)
```

This project has 12 fonts bundled with it, but you can use any OTF math
font. A python script is included that generates the `.plist` files 
required for an `.otf` font to work with `SwiftMath`.  If you generate
(and test) any other fonts please contribute them back to this project for
others to benefit.



Note: The `KpMath-Light`, `KpMath-Sans`, `Asana` fonts currently incorrectly
render very large radicals. It appears that the font files do
not properly define the offsets required to typeset these glyphs.  If
anyone can fix this, it would be greatly appreciated.

##### Text Color
The default color of the rendered equation is black. You can change
it to any other color as follows:

```swift
label.textColor = .red
```

It is also possible to set different colors for different parts of the
equation. Just access the `displayList` field and set the `textColor`
of the underlying displays of which you want to change the color. 

##### Custom Commands
You can define your own commands that are not already predefined. This is
similar to macros is LaTeX. To define your own command use:

```swift
MTMathAtomFactory.addLatexSymbol("lcm", value: MTMathAtomFactory.operator(withName: "lcm", limits: false))
```

This creates an `\lcm` command that can be used in the LaTeX.

##### Content Insets
The `MTMathUILabel` has `contentInsets` for finer control of placement of the
equation in relation to the view.

If you need to set it you can do as follows:

```swift
label.contentInsets = UIEdgeInsets(top: 0, left: 10, bottom: 0, right: 20)
```

##### Error handling

If the LaTeX text given to `MTMathUILabel` is
invalid or if it contains commands that aren't currently supported then
an error message will be displayed instead of the label.

This error can be programmatically retrieved as `label.error`. If you
prefer not to display anything then set:

```swift
label.displayErrorInline = true
```

## Future Enhancements

Note this is not a complete implementation of LaTeX math mode. There are
some important pieces that are missing and will be included in future
updates. This includes:

* Support for explicit big delimiters (bigl, bigr etc.)
* Addition of missing plain TeX commands 

## License

`SwiftMath` is available under the MIT license. See the [LICENSE](./LICENSE)
file for more info.

### Fonts
This distribution contains the following fonts. These fonts are
licensed as follows:
* Latin Modern Math: 
    [GUST Font License](GUST-FONT-LICENSE.txt)
* Tex Gyre Termes:
    [GUST Font License](GUST-FONT-LICENSE.txt)
* [XITS Math](https://github.com/khaledhosny/xits-math):
    [Open Font License](OFL.txt)
* [KpMath Light/KpMath Sans](http://scripts.sil.org/OFL):
    [SIL Open Font License](OFL.txt)
