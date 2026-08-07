SwiftUI Introspect
=================

[![CI Status Badge](https://github.com/siteline/swiftui-introspect/actions/workflows/ci.yml/badge.svg)](https://github.com/siteline/swiftui-introspect/actions/workflows/ci.yml)
[![Swift Version Compatibility Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fsiteline%2Fswiftui-introspect%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/siteline/swiftui-introspect)
[![Platform Compatibility Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fsiteline%2Fswiftui-introspect%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/siteline/swiftui-introspect)

SwiftUI Introspect allows you to get the underlying UIKit or AppKit element of a SwiftUI view.

For instance, with SwiftUI Introspect you can access `UITableView` to modify separators, or `UINavigationController` to customize the tab bar.

How it works
------------

SwiftUI Introspect works by adding an invisible `IntrospectionView` on top of the selected view, and an invisible "anchor" view underneath it, then looking through the UIKit/AppKit view hierarchy between the two to find the relevant view.

For instance, when introspecting a `ScrollView`...

```swift
ScrollView {
    Text("Item 1")
}
.introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { scrollView in
    // do something with UIScrollView
}
```

... it will:

1. Add marker views in front and behind `ScrollView`.
2. Traverse through all subviews between both marker views until a `UIScrollView` instance (if any) is found.

> [!IMPORTANT]
> Although this introspection method is very solid and unlikely to break in itself, future OS releases require explicit opt-in for introspection (`.iOS(.vXYZ)`), given potential differences in underlying UIKit/AppKit view types between major OS versions.

By default, the `.introspect` modifier acts directly on its _receiver_. This means calling `.introspect` from inside the view you're trying to introspect won't have any effect. However, there are times when this is not possible or simply too inflexible, in which case you **can** introspect an _ancestor_, but you must opt into this explicitly by overriding the introspection `scope`:

```swift
ScrollView {
    Text("Item 1")
        .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { scrollView in
            // do something with UIScrollView
        }
}
```

### Usage in production

SwiftUI Introspect is meant to be used in production. It does not use any private API. It only inspects the view hierarchy using publicly available methods. The library takes a defensive approach to inspecting the view hierarchy: there is no hard assumption that elements are laid out a certain way, there is no force-cast to UIKit/AppKit classes, and the `.introspect` modifier is simply ignored if UIKit/AppKit views cannot be found.

Install
-------

### Swift Package Manager

#### Xcode

<img width="656" src="https://github.com/siteline/swiftui-introspect/assets/2538074/d19c1dd3-9aa4-4e4f-a5a5-b2d6a5b9b927">

#### Package.swift

```swift
let package = Package(
    dependencies: [
        .package(url: "https://github.com/siteline/swiftui-introspect", from: "1.0.0"),
    ],
    targets: [
        .target(name: <#Target Name#>, dependencies: [
            .product(name: "SwiftUIIntrospect", package: "swiftui-introspect"),
        ]),
    ]
)
```

### CocoaPods

```ruby
pod 'SwiftUIIntrospect', '~> 1.0'
```

Introspection
-------------

### Implemented

- [`Button`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/buttontype)
- [`ColorPicker`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/colorpickertype)
- [`DatePicker`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickertype)
- [`DatePicker` with `.compact` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickerwithcompactstyletype)
- [`DatePicker` with `.field` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickerwithfieldstyletype)
- [`DatePicker` with `.graphical` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickerwithgraphicalstyletype)
- [`DatePicker` with `.stepperField` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickerwithstepperfieldstyletype)
- [`DatePicker` with `.wheel` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/datepickerwithwheelstyletype)
- [`Form`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/formtype)
- [`Form` with `.grouped` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/formwithgroupedstyletype)
- [`.fullScreenCover`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/fullScreenCovertype)
- [`List`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listtype)
- [`List` with `.bordered` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listwithborderedstyletype)
- [`List` with `.grouped` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listwithgroupedstyletype)
- [`List` with `.insetGrouped` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listwithinsetgroupedstyletype)
- [`List` with `.inset` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listwithinsetstyletype)
- [`List` with `.sidebar` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listwithsidebarstyletype)
- [`ListCell`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/listcelltype)
- [`Map`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/maptype)
- [`NavigationSplitView`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/navigationsplitviewtype)
- [`NavigationStack`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/navigationstacktype)
- [`NavigationView` with `.columns` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/NavigationViewWithColumnsStyleType)
- [`NavigationView` with `.stack` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/NavigationViewWithStackStyleType)
- [`PageControl`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/pagecontroltype)
- [`Picker` with `.menu` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/pickerwithmenustyletype)
- [`Picker` with `.segmented` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/pickerwithsegmentedstyletype)
- [`Picker` with `.wheel` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/pickerwithwheelstyletype)
- [`.popover`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/popovertype)
- [`ProgressView` with `.circular` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/progressviewwithcircularstyletype)
- [`ProgressView` with `.linear` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/progressviewwithlinearstyletype)
- [`ScrollView`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/scrollviewtype)
- [`.searchable`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/searchfieldtype)
- [`SecureField`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/securefieldtype)
- [`.sheet`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/sheettype)
- [`Slider`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/slidertype)
- [`Stepper`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/steppertype)
- [`Table`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/tabletype)
- [`TabView`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/tabviewtype)
- [`TabView` with `.page` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/TabViewWithPageStyleType)
- [`TextEditor`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/texteditortype)
- [`TextField`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/textfieldtype)
- [`TextField` with `.vertical` axis](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/TextFieldWithVerticalAxisType)
- [`Toggle`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/toggletype)
- [`Toggle` with `button` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/togglewithbuttonstyletype)
- [`Toggle` with `checkbox` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/togglewithcheckboxstyletype)
- [`Toggle` with `switch` style](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/togglewithswitchstyletype)
- [`VideoPlayer`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/videoplayertype)
- [`View`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/viewtype)
- [`ViewController`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/viewcontrollertype)
- [`Window`](https://swiftpackageindex.com/siteline/swiftui-introspect/main/documentation/swiftuiintrospect/windowtype)

**Missing an element?** Please [start a discussion](https://github.com/siteline/swiftui-introspect/discussions/new?category=ideas). As a temporary solution, you can [implement your own introspectable view type](#implement-your-own-introspectable-type).

### Cannot implement

SwiftUI | Affected Frameworks | Why
--- | --- | ---
Text | UIKit, AppKit | Not a UILabel / NSLabel
Image | UIKit, AppKit | Not a UIImageView / NSImageView
Button | UIKit | Not a UIButton

Examples
--------

### List

```swift
List {
    Text("Item")
}
.introspect(.list, on: .iOS(.v13, .v14, .v15)) { tableView in
    tableView.backgroundView = UIView()
    tableView.backgroundColor = .cyan
}
.introspect(.list, on: .iOS(.v16, .v17, .v18)) { collectionView in
    collectionView.backgroundView = UIView()
    collectionView.subviews.dropFirst(1).first?.backgroundColor = .cyan
}
```

### ScrollView

```swift
ScrollView {
    Text("Item")
}
.introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { scrollView in
    scrollView.backgroundColor = .red
}
```

### NavigationView

```swift
NavigationView {
    Text("Item")
}
.navigationViewStyle(.stack)
.introspect(.navigationView(style: .stack), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { navigationController in
    navigationController.navigationBar.backgroundColor = .cyan
}
```

### TextField

```swift
TextField("Text Field", text: <#Binding<String>#>)
    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { textField in
        textField.backgroundColor = .red
    }
```

Advanced usage
--------------

### Implement your own introspectable type

**Missing an element?** Please [start a discussion](https://github.com/siteline/swiftui-introspect/discussions/new?category=ideas).

In case SwiftUI Introspect (unlikely) doesn't support the SwiftUI element that you're looking for, you can implement your own introspectable type.

For example, here's how the library implements the introspectable `TextField` type:

```swift
import SwiftUI
@_spi(Advanced) import SwiftUIIntrospect

public struct TextFieldType: IntrospectableViewType {}

extension IntrospectableViewType where Self == TextFieldType {
    public static var textField: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<TextFieldType, UITextField> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<TextFieldType, UITextField> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<TextFieldType, UITextField> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<TextFieldType, NSTextField> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
```

### Introspect on future platform versions

By default, introspection applies per specific platform version. This is a sensible default for maximum predictability in regularly maintained codebases, but it's not always a good fit for e.g. library developers who may want to cover as many future platform versions as possible in order to provide the best chance for long-term future functionality of their library without regular maintenance.

For such cases, SwiftUI Introspect offers range-based platform version predicates behind the Advanced SPI:

```swift
import SwiftUI
@_spi(Advanced) import SwiftUIIntrospect

struct ContentView: View {
    var body: some View {
        ScrollView {
            // ...
        }
        .introspect(.scrollView, on: .iOS(.v13...)) { scrollView in
            // ...
        }
    }
}
```

Bear in mind this should be used cautiously, and with full knowledge that any future OS version might break the expected introspection types unless explicitly available. For instance, if in the example above hypothetically iOS 19 stops using UIScrollView under the hood, the customization closure will never be called on said platform.

### Keep instances outside the customize closure

Sometimes, you might need to keep your introspected instance around for longer than the customization closure lifetime. In such cases, `@State` is not a good option because it produces retain cycles. Instead, SwiftUI Introspect offers a `@Weak` property wrapper behind the Advanced SPI:

```swift
import SwiftUI
@_spi(Advanced) import SwiftUIIntrospect

struct ContentView: View {
    @Weak var scrollView: UIScrollView?

    var body: some View {
        ScrollView {
            // ...
        }
        .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { scrollView in
            self.scrollView = scrollView
        }
    }
}
```

Community projects
------------------

Here's a list of open source libraries powered by the SwiftUI Introspect library:

<a href="https://github.com/paescebu/CustomKeyboardKit">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=paescebu&repo=CustomKeyboardKit" />
</a>

<a href="https://github.com/davdroman/swiftui-navigation-transitions">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=davdroman&repo=swiftui-navigation-transitions" />
</a>

If you're working on a library built on SwiftUI Introspect or know of one, feel free to submit a PR adding it to the list.
