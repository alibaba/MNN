#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Form` type in SwiftUI, with `.grouped` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         Form {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .formStyle(.grouped)
///         .introspect(.form(style: .grouped), on: .iOS(.v16, .v17, .v18)) {
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
///         Form {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .formStyle(.grouped)
///         .introspect(.form(style: .grouped), on: .tvOS(.v16, .v17, .v18)) {
///             print(type(of: $0)) // UITableView
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
///         Form {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .formStyle(.grouped)
///         .introspect(.form(style: .grouped), on: .macOS(.v13, .v14, .v15)) {
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
///         Form {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .formStyle(.grouped)
///         .introspect(.form(style: .grouped), on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
public struct FormWithGroupedStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case grouped
    }
}

extension IntrospectableViewType where Self == FormWithGroupedStyleType {
    public static func form(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<FormWithGroupedStyleType, UITableView> {
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on iOS 14")
    public static let v14 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on iOS 15")
    public static let v15 = Self.unavailable()
}

extension iOSViewVersion<FormWithGroupedStyleType, UICollectionView> {
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<FormWithGroupedStyleType, UITableView> {
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on tvOS 14")
    public static let v14 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on tvOS 15")
    public static let v15 = Self.unavailable()
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<FormWithGroupedStyleType, UICollectionView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<FormWithGroupedStyleType, NSScrollView> {
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on macOS 11")
    public static let v11 = Self.unavailable()
    @available(*, unavailable, message: ".formStyle(.grouped) isn't available on macOS 12")
    public static let v12 = Self.unavailable()
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
