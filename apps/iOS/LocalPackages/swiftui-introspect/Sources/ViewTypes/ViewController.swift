#if !os(watchOS)
import SwiftUI

/// An abstract representation of the receiving SwiftUI view's view controller,
/// or the closest ancestor view controller if missing.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         NavigationView {
///             Text("Root").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///                 .introspect(.viewController, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                     print(type(of: $0)) // some subclass of UIHostingController
///                 }
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.viewController, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
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
///             Text("Root").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///                 .introspect(.viewController, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                     print(type(of: $0)) // some subclass of UIHostingController
///                 }
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.viewController, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
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
///             Text("Root").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
///                 .introspect(.viewController, on: .visionOS(.v1, .v2)) {
///                     print(type(of: $0)) // some subclass of UIHostingController
///                 }
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.viewController, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UINavigationController
///         }
///     }
/// }
/// ```
public struct ViewControllerType: IntrospectableViewType {
    public var scope: IntrospectionScope { [.receiver, .ancestor] }
}

extension IntrospectableViewType where Self == ViewControllerType {
    public static var viewController: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ViewControllerType, UIViewController> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<ViewControllerType, UIViewController> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ViewControllerType, UIViewController> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#endif
#endif
