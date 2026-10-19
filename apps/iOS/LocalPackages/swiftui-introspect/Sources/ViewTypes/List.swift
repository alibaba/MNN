#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `List` type in SwiftUI.
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
///         .introspect(.list, on: .iOS(.v13, .v14, .v15)) {
///             print(type(of: $0)) // UITableView
///         }
///         .introspect(.list, on: .iOS(.v16, .v17, .v18)) {
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
///         List {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .introspect(.list, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
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
///         List {
///             Text("Item 1")
///             Text("Item 2")
///             Text("Item 3")
///         }
///         .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
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
///         .introspect(.list, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
public struct ListType: IntrospectableViewType {
    public enum Style: Sendable {
        case plain
    }
}

extension IntrospectableViewType where Self == ListType {
    public static var list: Self { .init() }
    public static func list(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ListType, UITableView> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}

extension iOSViewVersion<ListType, UICollectionView> {
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<ListType, UITableView> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ListType, UICollectionView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<ListType, NSTableView> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
