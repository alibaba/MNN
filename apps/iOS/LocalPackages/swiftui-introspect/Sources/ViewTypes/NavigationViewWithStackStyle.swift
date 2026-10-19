#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `NavigationView` type in SwiftUI, with `.stack` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         NavigationView {
///             Text("Root")
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.navigationView(style: .stack), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UINavigationController
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
///         NavigationView {
///             Text("Root")
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.navigationView(style: .stack), on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UINavigationController
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
///         NavigationView {
///             Text("Root")
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.navigationView(style: .stack), on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UINavigationController
///         }
///     }
/// }
/// ```
public struct NavigationViewWithStackStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case stack
    }
}

extension IntrospectableViewType where Self == NavigationViewWithStackStyleType {
    public static func navigationView(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<NavigationViewWithStackStyleType, UINavigationController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UINavigationController> {
        .default.withAncestorSelector { $0.navigationController }
    }
}

extension tvOSViewVersion<NavigationViewWithStackStyleType, UINavigationController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UINavigationController> {
        .default.withAncestorSelector { $0.navigationController }
    }
}

extension visionOSViewVersion<NavigationViewWithStackStyleType, UINavigationController> {
    public static let v1 = Self(for: .v1, selector: selector)
    public static let v2 = Self(for: .v2, selector: selector)

    private static var selector: IntrospectionSelector<UINavigationController> {
        .default.withAncestorSelector { $0.navigationController }
    }
}
#endif
#endif
