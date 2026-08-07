#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `TabView` type in SwiftUI, with `.page` style.
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
///         .introspect(.tabView(style: .page), on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UICollectionView
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
///         .introspect(.tabView(style: .page), on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UICollectionView
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
///         .introspect(.tabView(style: .page), on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
public struct TabViewWithPageStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case page
    }
}

#if !os(macOS)
extension IntrospectableViewType where Self == TabViewWithPageStyleType {
    public static func tabView(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<TabViewWithPageStyleType, UICollectionView> {
    @available(*, unavailable, message: ".tabViewStyle(.page) isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<TabViewWithPageStyleType, UICollectionView> {
    @available(*, unavailable, message: ".tabViewStyle(.page) isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<TabViewWithPageStyleType, UICollectionView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#endif
#endif
#endif
