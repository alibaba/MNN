#if !os(watchOS)
import SwiftUI

/// An abstract representation of the page control type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         TabView {
///             Text("Page 1").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///             Text("Page 2").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.blue)
///         }
///         .tabViewStyle(.page(indexDisplayMode: .always))
///         .introspect(.pageControl, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UIPageControl
///         }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         TabView {
///             Text("Page 1").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///             Text("Page 2").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.blue)
///         }
///         .tabViewStyle(.page(indexDisplayMode: .always))
///         .introspect(.pageControl, on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UIPageControl
///         }
///     }
/// }
/// ```
///
/// ### macOS
///
/// Not available.
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         TabView {
///             Text("Page 1").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///             Text("Page 2").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.blue)
///         }
///         .tabViewStyle(.page(indexDisplayMode: .always))
///         .introspect(.pageControl, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UIPageControl
///         }
///     }
/// }
/// ```
public struct PageControlType: IntrospectableViewType {}

extension IntrospectableViewType where Self == PageControlType {
    public static var pageControl: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<PageControlType, UIPageControl> {
    @available(*, unavailable, message: ".tabViewStyle(.page) isn't available on iOS 13")
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<PageControlType, UIPageControl> {
    @available(*, unavailable, message: ".tabViewStyle(.page) isn't available on tvOS 13")
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<PageControlType, UIPageControl> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#endif
#endif
