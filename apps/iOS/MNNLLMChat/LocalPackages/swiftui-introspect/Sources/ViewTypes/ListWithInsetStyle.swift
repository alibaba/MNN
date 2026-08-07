#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `List` type in SwiftUI, with `.inset` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         List {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .listStyle(.inset)
///         .introspect(.list(style: .inset), on: .iOS(.v14, .v15)) {
///             print(type(of: $0)) // UITableView
///         }
///         .introspect(.list(style: .inset), on: .iOS(.v16, .v17, .v18)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// Not available.
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         List {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .listStyle(.inset)
///         .introspect(.list(style: .inset), on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSTableView
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
///         List {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .listStyle(.inset)
///         .introspect(.list(style: .inset), on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
public struct ListWithInsetStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case inset
    }
}

#if !os(tvOS)
extension IntrospectableViewType where Self == ListWithInsetStyleType {
    public static func list(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ListWithInsetStyleType, UITableView> {
    @available(*, unavailable, message: ".listStyle(.inset) isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}

extension iOSViewVersion<ListWithInsetStyleType, UICollectionView> {
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ListWithInsetStyleType, UICollectionView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<ListWithInsetStyleType, NSTableView> {
    @available(*, unavailable, message: ".listStyle(.inset) isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
