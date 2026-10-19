<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/header-dark.png"><img src="https://raw.githubusercontent.com/exyte/media/master/common/header-light.png"></picture></a>

<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/our-site-dark.png" width="80" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/our-site-light.png" width="80" height="16"></picture></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://twitter.com/exyteHQ"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/twitter-dark.png" width="74" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/twitter-light.png" width="74" height="16">
</picture></a> <a href="https://exyte.com/contacts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-dark.png" width="128" height="24" align="right"><img src="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-light.png" width="128" height="24" align="right"></picture></a>

<img src="https://raw.githubusercontent.com/exyte/media/master/FloatingButton/demo.gif" width="480" />

<p><h1 align="left">FloatingButton</h1></p>

<p><h4>Easily customizable floating button menu created with SwiftUI</h4></p>

![](https://img.shields.io/github/v/tag/exyte/FloatingButton?label=Version)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FFloatingButton%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/exyte/FloatingButton)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FFloatingButton%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/exyte/FloatingButton)
[![SPM](https://img.shields.io/badge/SPM-Compatible-brightgreen.svg)](https://swiftpackageindex.com/exyte/FloatingButton)
[![Cocoapods](https://img.shields.io/badge/Cocoapods-Deprecated%20after%201.3.0-yellow.svg)](https://cocoapods.org/pods/FloatingButton)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)

# Usage

1. Create main button view and a number of submenu buttons — both should be cast to `AnyView` type.
2. Pass both to `FloatingButton` constructor:

   ```swift
   FloatingButton(mainButtonView: mainButton, buttons: buttons)
   ```
3. You may also pass a binding which will determine if the menu is currently open. You may use this to close the menu on any submenu button tap for example. 
```swift
FloatingButton(mainButtonView: mainButton, buttons: buttons, isOpen: $isOpen)
```
4. Chain `.straight()` or `.circle()` to specify desired menu type.
5. Chain whatever you like afterwards. For example:
    ```swift
    FloatingButton(mainButtonView: mainButton, buttons: textButtons)
        .straight()
        .direction(.top)
        .alignment(.left)
        .spacing(10)
        .initialOffset(x: -1000)
        .animation(.spring())

    FloatingButton(mainButtonView: mainButton2, buttons: buttonsImage.dropLast())
        .circle()
        .startAngle(3/2 * .pi)
        .endAngle(2 * .pi)
        .radius(70)
        .layoutDirection(.counterClockwise)
    ```

### Universal options
`spacing` - space between submenu buttons  
`initialScaling` - size multiplyer for submenu buttons when the menu is closed  
`initialOffset` - offset for submenu buttons when the menu is closed  
`initialOpacity` - opacity for submenu buttons when the menu is closed  
`animation` - custom SwiftUI animation like `Animation.easeInOut()` or `Animation.spring()`  
`delays` - delay for each submenu button's animation start
    - you can pass array of delays - one for each element
    - or you can pass `delayDelta` - then this same delay will be used for each element
`mainZStackAlignment` - main button and submenu buttons are contained in one ZStack (not an overlay so the menu has a correct size), you can change this ZStack's alignment with this parameter
`inverseZIndex` - inverse zIndex of mainButton and the children. Use, for example, if you have a negative spacing and want to change the order
`wholeMenuSize` - pass CGSize binding to get updates of menu's size. Menu's size includes main button frame and all of elements' frames
`menuButtonsSize` - pass CGSize binding to get updates of combined menu elements' size

### Straight menu only options

`direction` - position of submenu buttons relative to main menu button  
`alignment` - alignment of submenu buttons relative to main menu button 

### Circle only options

`startAngle`  
`endAngle`  
`radius` - distance between center of main button and centers of submenu buttons  
`layoutDirection` - changes the button layout direction from the startAngle to the endAngle

## Examples

To try the FloatingButton examples:
- Clone the repo `https://github.com/exyte/FloatingButton.git`
- Open `FloatingButtonExample.xcodeproj` in the Xcode
- Try it!

## Installation

### [Swift Package Manager](https://swift.org/package-manager/)

```swift
dependencies: [
    .package(url: "https://github.com/exyte/FloatingButton.git")
]
```

## Requirements

* iOS 14.0+ / macOS 11.0+ / watchOS 7.0+
* Xcode 12+

## Our other open source SwiftUI libraries
[PopupView](https://github.com/exyte/PopupView) - Toasts and popups library    
[AnchoredPopup](https://github.com/exyte/AnchoredPopup) - Anchored Popup grows "out" of a trigger view (similar to Hero animation)    
[Grid](https://github.com/exyte/Grid) - The most powerful Grid container    
[ScalingHeaderScrollView](https://github.com/exyte/ScalingHeaderScrollView) - A scroll view with a sticky header which shrinks as you scroll    
[AnimatedTabBar](https://github.com/exyte/AnimatedTabBar) - A tabbar with a number of preset animations   
[MediaPicker](https://github.com/exyte/mediapicker) - Customizable media picker     
[Chat](https://github.com/exyte/chat) - Chat UI framework with fully customizable message cells, input view, and a built-in media picker  
[OpenAI](https://github.com/exyte/OpenAI) Wrapper lib for [OpenAI REST API](https://platform.openai.com/docs/api-reference/introduction)    
[AnimatedGradient](https://github.com/exyte/AnimatedGradient) - Animated linear gradient     
[ConcentricOnboarding](https://github.com/exyte/ConcentricOnboarding) - Animated onboarding flow    
[ActivityIndicatorView](https://github.com/exyte/ActivityIndicatorView) - A number of animated loading indicators    
[ProgressIndicatorView](https://github.com/exyte/ProgressIndicatorView) - A number of animated progress indicators    
[FlagAndCountryCode](https://github.com/exyte/FlagAndCountryCode) - Phone codes and flags for every country    
[SVGView](https://github.com/exyte/SVGView) - SVG parser    
[LiquidSwipe](https://github.com/exyte/LiquidSwipe) - Liquid navigation animation    

