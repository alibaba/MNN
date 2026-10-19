#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `NavigationSplitView` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         NavigationSplitView {
///             Text("Root")
///         } detail: {
///             Text("Detail")
///         }
///         .introspect(.navigationSplitView, on: .iOS(.v16, .v17, .v18)) {
///             print(type(of: $0)) // UISplitViewController
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
///         NavigationSplitView {
///             Text("Root")
///         } detail: {
///             Text("Detail")
///         }
///         .introspect(.navigationSplitView, on: .tvOS(.v16, .v17)) {
///             print(type(of: $0)) // UINavigationController
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
///         NavigationSplitView {
///             Text("Root")
///         } detail: {
///             Text("Detail")
///         }
///         .introspect(.navigationSplitView, on: .macOS(.v13, .v14, .v15)) {
///             print(type(of: $0)) // NSSplitView
///         }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         NavigationSplitView {
///             Text("Root")
///         } detail: {
///             Text("Detail")
///         }
///         .introspect(.navigationSplitView, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UISplitViewController
///         }
///     }
/// }
/// ```
public struct NavigationSplitViewType: IntrospectableViewType {}

extension IntrospectableViewType where Self == NavigationSplitViewType {
    public static var navigationSplitView: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<NavigationSplitViewType, UISplitViewController> {
    @available(*, unavailable, message: "NavigationSplitView isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on iOS 14")
    public static let v14 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on iOS 15")
    public static let v15 = Self.unavailable()

    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UISplitViewController> {
        .default.withAncestorSelector { $0.splitViewController }
    }
}

extension tvOSViewVersion<NavigationSplitViewType, UINavigationController> {
    @available(*, unavailable, message: "NavigationSplitView isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on tvOS 14")
    public static let v14 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on tvOS 15")
    public static let v15 = Self.unavailable()

    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    @available(*, unavailable, message: "NavigationSplitView isn't backed by UIKit since tvOS 18")
    public static let v18 = Self.unavailable()

    private static var selector: IntrospectionSelector<UINavigationController> {
        .default.withAncestorSelector { $0.navigationController }
    }
}

extension visionOSViewVersion<NavigationSplitViewType, UISplitViewController> {
    public static let v1 = Self(for: .v1, selector: selector)
    public static let v2 = Self(for: .v2, selector: selector)

    private static var selector: IntrospectionSelector<UISplitViewController> {
        .default.withAncestorSelector { $0.splitViewController }
    }
}
#elseif canImport(AppKit)
extension macOSViewVersion<NavigationSplitViewType, NSSplitView> {
    @available(*, unavailable, message: "NavigationSplitView isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on macOS 11")
    public static let v11 = Self.unavailable()
    @available(*, unavailable, message: "NavigationSplitView isn't available on macOS 12")
    public static let v12 = Self.unavailable()

    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
