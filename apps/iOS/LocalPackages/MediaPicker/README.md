<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/header-dark.png"><img src="https://raw.githubusercontent.com/exyte/media/master/common/header-light.png"></picture></a>

<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/our-site-dark.png" width="80" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/our-site-light.png" width="80" height="16"></picture></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://twitter.com/exyteHQ"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/twitter-dark.png" width="74" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/twitter-light.png" width="74" height="16">
</picture></a> <a href="https://exyte.com/contacts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-dark.png" width="128" height="24" align="right"><img src="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-light.png" width="128" height="24" align="right"></picture></a>

<img src="https://raw.githubusercontent.com/exyte/media/master/MediaPicker/banner.png" />

<p><h1 align="left">Media Picker</h1></p>

<p><h4>SwiftUI library for a customizable media picker.</h4></p>

![](https://img.shields.io/github/v/tag/exyte/MediaPicker?label=Version)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FMediaPicker%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/exyte/MediaPicker)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FMediaPicker%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/exyte/MediaPicker)
[![SPM Compatible](https://img.shields.io/badge/SwiftPM-Compatible-brightgreen.svg)](https://swiftpackageindex.com/exyte/MediaPicker)
[![Cocoapods Compatible](https://img.shields.io/badge/cocoapods-Compatible-brightgreen.svg)](https://cocoapods.org/pods/ExyteMediaPicker)
[![Carthage Compatible](https://img.shields.io/badge/Carthage-compatible-brightgreen.svg?style=flat)](https://github.com/Carthage/Carthage)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)

# Features
* Photo and video picker
* Single and multiple selection
* Full-screen view
* Live photo preview & capture
* Full customization

# MediaPicker vs PhotosPicker
* The iOS 16 PhotosPicker only provides you with a button, while this library gives you the whole view, meaning you can build it into you own screens and customize it as you see fit. 
* PhotosPicker only lets you pick photos from the library - no camera. 
* MediaPicker provides a default looking library picker, with ability to manage albums, and also a camera view to take photos/video

## !SPM Renaming!
SPM package is now called `ExyteMediaPicker` instead of `MediaPicker`, sorry for any inconveniences.

# Usage
1. Add a binding Bool to control the picker presentation state.
2. Add Media array to save selection (`[Media]`).
3. Initialize the media picker and present it however you like - for example, using the .sheet modifier
```swift
.sheet(isPresented: $showMediaPicker) {
    MediaPicker(
        isPresented: $showMediaPicker,
        onChange: { medias = $0 }
    )
}
```

## Media model
The lbrary will return an array of `Media` structs for you to use as you see fit. It has the following fields and methods (all the methods use async/await API):
- `type` - .image or .video
- `duration` - nil for .image
- `getURL()` - returns `URL` to media (automatically stores in temporary directory if needed)
- `getThumbnailURL()` - returns `URL` to media's thumbnail (for image just returns the image itself)
- `getData()` - returns media's `Data` representation
- `getThumbnailData()` - returns media's thumbnail `Data` representation

## Modes
This library lets you use both photo library and camera

### Photos grid
Default photos grid screen has a standard header which contains the 'Done' and 'Cancel' buttons, and a simple switcher between Photos and Albums. Use it for a basic out-of-the box picker (see default picker in example project for an usage example). This can be customized (see "Init - view builders" section)  

### Camera
After making one photo, you see a preview of it and a little plus icon, by tapping it you return back to camera mode and can continue making as many photos as you like. Press "Done" once you're finished and you will be able to scroll through all the photos you've taken before confirming you'd like to use them. This preview screen of photos you've taken can also be customized (see "Init - view builders" section)     

## Init - required parameters
`isPresented` - a binding to determine whether the picker should be displayed or not   
`onChange` - a closure that returns the selected media every time the selection changes

### Init - optional view builders
You can pass 1-3 view builders in order to add your own buttons and other elements to media picker screens. You can pass all, some or none of these when creating your `MediaPicker` (see the custom picker in the example project for usage example). First screen you can customize is default photos grid view. Pass `albumSelectionBuilder` closure like this to replace the standard one with your own view:
```swift
MediaPicker(
    isPresented: $isPresented,
    onChange: { selectedMedia = $0 },
    albumSelectionBuilder: { defaultHeaderView, albumSelectionView, isInFullscreen in
        VStack {
            if !isInFullscreen {
                defaultHeaderView
            }
            albumSelectionView
            Spacer()
            footerView
        }
        .background(Color.black)
    }
)
```

`albumSelectionBuilder` gives you two views to work with:
- `defaultHeaderView` - a default looking `header` with photos/albums mode switcher
- `albumSelectionView` - the photos grid itself
- `isInFullscreen` - is fullscreen photo details screen displayed. Use for example to hide the header while in fullscreen mode.

The second customizable screen is the one you see after taking a photo. Pass `cameraSelectionBuilder` like this:
```swift
MediaPicker(
    isPresented: $isPresented,
    onChange: { selectedMedia = $0 },
    cameraSelectionBuilder: { addMoreClosure, cancelClosure, cameraSelectionView in
        VStack {
            HStack {
                Spacer()
                Button("Done", action: { isPresented = false })
            }
            cameraSelectionView
            HStack {
                Button("Cancel", action: cancelClosure)
                Spacer()
                Button(action: addMoreClosure) {
                    Text("Take more photos")
                }
            }
        }
    }
)
```
`cameraSelectionBuilder` gives you these parameters:
- `addMoreClosure` - you can call this closure on tap of your own button, it will work same as default plus icon on camera selection preview screen
- `cancelClosure` - show confirmation and return to photos grid screen if confirmed
- `cameraSelectionView` - swipable camera photos preview collection itself

The last one is live camera screen

```swift
MediaPicker(
    isPresented: $isPresented,
    onChange: { selectedMedia = $0 },
    cameraViewBuilder: { cameraSheetView, cancelClosure, showPreviewClosure, takePhotoClosure, startVideoCaptureClosure, stopVideoCaptureClosure, toggleFlash, flipCamera in
        cameraSheetView
            .overlay(alignment: .topLeading) {
                HStack {
                    Button("Cancel") { cancelClosure() }
                        .foregroundColor(Color("CustomPurple"))
                    Spacer()
                    Button("Done") { showPreviewClosure() }
                        .foregroundColor(Color("CustomPurple"))
                }
                .padding()
            }
            .overlay(alignment: .bottom) {
                HStack {
                    Button("Take photo") { takePhotoClosure() }
                        .greenButtonStyle()
                    Button(videoIsBeingRecorded ? "Stop video capture" : "Capture video") {
                        videoIsBeingRecorded ? stopVideoCaptureClosure() : startVideoCaptureClosure()
                        videoIsBeingRecorded.toggle()
                    }
                    .greenButtonStyle()
                }
                .padding()
            }
    }
)
```

`cameraViewBuilder` live camera capture view and a lot of closures to do with as you please:
- `cameraSheetView` - live camera capture view
- `cancelClosure` - if you want to display "are you sure" before closing
- `showPreviewClosure` - shows preview of taken photos
- `cancelClosure` - if you want to display "are you sure" before closing
- `startVideoCaptureClosure` - starts video capture, you'll need a bollean varialbe to track recording state
- `stopVideoCaptureClosure` - stops video capture
- `toggleFlash` - flash off/on
- `flipCamera` - camera back/front

## Available modifiers
`showLiveCameraCell` - show live camera feed cell in the top left corner of the gallery grid     
`mediaSelectionType` - limit displayed media type: .photo, .video or both   
`mediaSelectionStyle` - a way to display selected/unselected media state: a counter or a simple checkmark   
`mediaSelectionLimit` - the maximum selection quantity allowed, 'nil' for unlimited selection   
`showFullscreenPreview` - if true - tap on media opens fullscreen preview, if false - tap on image immediately selects this image and closes the picker

### Available modifiers - filtering
`applyFilter((Media) async -> Media?)` - pass a closure to apply to each of medias individually. Closures's return type is `Media?`: return `Media` the closure gives to you if you want it to be displayed on photo grid, or `nil` if you want to exclude it. The code you apply to each media can be asyncronous (using async/await syntactics, check out `FilterMediaPicker` in example project)
`applyFilter(([Media]) async -> [Media])` - same but apply the closure to whole medias array. Can also be used for reodering. 

### Available modifiers - screen rotation
If your app restricts screen rotation, you can skip this section.

We recommend locking orientation for MediaPicker, because default rotation animations don't look good on the camera screen. At the moment SwiftUI doesn't provide a way of locking screen orientation, so the library has an initializer with an `orientationHandler` parameter - a closure that is called when you enter/leave the camera screen inside MediaPicker. In this closure you need to use AppDelegate to lock/unlock the rotation - see example project for implementation.

### Available modifiers: managing albums  
`albums` - a list of user's albums (like in Photos app), if you want to display them differently than `showingDefaultHeader` does.           
`pickerMode` - set this if you don't plan to use the default header. Available options are:     
    * .photos - displays the default photos grid      
    * .albums - displays a list of albums with one preview photo for each     
    * .album(Album) - displays one album     
    * .camera - shows a fullscreen cover camera sheet
    * .cameraSelection - displays a preview of photos taken with camera 
(see the custom picker in the example project for implementation)

<img src="https://raw.githubusercontent.com/exyte/media/master/MediaPicker/2.jpg" width="250"/>

### Available modifiers: theme  
`mediaPickerTheme` - color settings. Example usage (see `MediaPickerTheme` for all available settings):    
  ```swift
MediaPicker(...)
    .mediaPickerTheme(
        main: .init(
            background: .black
        ),
        selection: .init(
            emptyTint: .white,
            emptyBackground: .black.opacity(0.25),
            selectedTint: Color("CustomPurple")
        )
    )
  ```
Here is an example of how you can customize colors and elements to create a custom looking picker:  

<img src="https://raw.githubusercontent.com/exyte/media/master/MediaPicker/1.jpg" width="250"/>

## Examples

To try out the MediaPicker examples:
- Clone the repo `https://github.com/exyte/MediaPicker.git`
- Open `Examples/Examples.xcworkspace` in the Xcode
- Run it!

## Installation

### [Swift Package Manager](https://swift.org/package-manager/)

```swift
dependencies: [
    .package(url: "https://github.com/exyte/ExyteMediaPicker.git")
]
```

### CocoaPods

```ruby
pod 'ExyteMediaPicker'
```

### Carthage

```ogdl
github "Exyte/MediaPicker"
```

## Requirements

* iOS 16+
* Xcode 13+ 

## Our other open source SwiftUI libraries
[PopupView](https://github.com/exyte/PopupView) - Toasts and popups library    
[Grid](https://github.com/exyte/Grid) - The most powerful Grid container    
[ScalingHeaderScrollView](https://github.com/exyte/ScalingHeaderScrollView) - A scroll view with a sticky header which shrinks as you scroll    
[AnimatedTabBar](https://github.com/exyte/AnimatedTabBar) - A tabbar with a number of preset animations   
[Chat](https://github.com/exyte/chat) - Chat UI framework with fully customizable message cells, input view, and a built-in media picker  
[OpenAI](https://github.com/exyte/OpenAI) Wrapper lib for [OpenAI REST API](https://platform.openai.com/docs/api-reference/introduction)    
[AnimatedGradient](https://github.com/exyte/AnimatedGradient) - Animated linear gradient     
[ConcentricOnboarding](https://github.com/exyte/ConcentricOnboarding) - Animated onboarding flow    
[FloatingButton](https://github.com/exyte/FloatingButton) - Floating button menu    
[ActivityIndicatorView](https://github.com/exyte/ActivityIndicatorView) - A number of animated loading indicators    
[ProgressIndicatorView](https://github.com/exyte/ProgressIndicatorView) - A number of animated progress indicators    
[FlagAndCountryCode](https://github.com/exyte/FlagAndCountryCode) - Phone codes and flags for every country    
[SVGView](https://github.com/exyte/SVGView) - SVG parser    
[LiquidSwipe](https://github.com/exyte/LiquidSwipe) - Liquid navigation animation    
