#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `TabView` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         TabView {
///             Text("Tab 1").tabItem { Text("Tab 1") }
///             Text("Tab 2").tabItem { Text("Tab 2") }
///         }
///         .introspect(.tabView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UITabBarController
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
///             Text("Tab 1").tabItem { Text("Tab 1") }
///             Text("Tab 2").tabItem { Text("Tab 2") }
///         }
///         .introspect(.tabView, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UITabBarController
///         }
///     }
/// }
/// ```
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         TabView {
///             Text("Tab 1").tabItem { Text("Tab 1") }
///             Text("Tab 2").tabItem { Text("Tab 2") }
///         }
///         .introspect(.tabView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSTabView
///         }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct TabViewType: IntrospectableViewType {}

#if !os(visionOS)
extension IntrospectableViewType where Self == TabViewType {
    public static var tabView: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<TabViewType, UITabBarController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    @MainActor
    private static var selector: IntrospectionSelector<UITabBarController> {
        .default.withAncestorSelector { $0.tabBarController }
    }
}

extension tvOSViewVersion<TabViewType, UITabBarController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    @MainActor
    private static var selector: IntrospectionSelector<UITabBarController> {
        .default.withAncestorSelector { $0.tabBarController }
    }
}
#elseif canImport(AppKit)
extension macOSViewVersion<TabViewType, NSTabView> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
