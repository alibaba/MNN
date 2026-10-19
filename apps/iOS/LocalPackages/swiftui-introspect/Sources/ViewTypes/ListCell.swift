#if !os(watchOS)
import SwiftUI

/// An abstract representation of a `List` cell type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         List {
///             ForEach(1...3, id: \.self) { int in
///                 Text("Item \(int)")
///                     .introspect(.listCell, on: .iOS(.v13, .v14, .v15)) {
///                         print(type(of: $0)) // UITableViewCell
///                     }
///                     .introspect(.listCell, on: .iOS(.v16, .v17, .v18)) {
///                         print(type(of: $0)) // UICollectionViewCell
///                     }
///             }
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
///             ForEach(1...3, id: \.self) { int in
///                 Text("Item \(int)")
///                     .introspect(.listCell, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                         print(type(of: $0)) // UITableViewCell
///                     }
///             }
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
///             ForEach(1...3, id: \.self) { int in
///                 Text("Item \(int)")
///                     .introspect(.listCell, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///                         print(type(of: $0)) // NSTableCellView
///                     }
///             }
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
///             ForEach(1...3, id: \.self) { int in
///                 Text("Item \(int)")
///                     .introspect(.listCell, on: .visionOS(.v1, .v2)) {
///                         print(type(of: $0)) // UICollectionViewCell
///                     }
///             }
///         }
///     }
/// }
/// ```
public struct ListCellType: IntrospectableViewType {
    public var scope: IntrospectionScope { .ancestor }
}

extension IntrospectableViewType where Self == ListCellType {
    public static var listCell: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ListCellType, UITableViewCell> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}

extension iOSViewVersion<ListCellType, UICollectionViewCell> {
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<ListCellType, UITableViewCell> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ListCellType, UICollectionViewCell> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<ListCellType, NSTableCellView> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
