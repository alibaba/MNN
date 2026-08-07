#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `ScrollView` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         ScrollView {
///             Text("Item")
///         }
///         .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UIScrollView
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
///         ScrollView {
///             Text("Item")
///         }
///         .introspect(.scrollView, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UIScrollView
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
///         ScrollView {
///             Text("Item")
///         }
///         .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSScrollView
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
///         ScrollView {
///             Text("Item")
///         }
///         .introspect(.scrollView, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UIScrollView
///         }
///     }
/// }
/// ```
public struct ScrollViewType: IntrospectableViewType {}

extension IntrospectableViewType where Self == ScrollViewType {
    public static var scrollView: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ScrollViewType, UIScrollView> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<ScrollViewType, UIScrollView> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ScrollViewType, UIScrollView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<ScrollViewType, NSScrollView> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
