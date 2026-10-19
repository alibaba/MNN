#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Table` type in SwiftUI, with any style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     struct Purchase: Identifiable {
///         let id = UUID()
///         let price: Decimal
///     }
///
///     var body: some View {
///         Table(of: Purchase.self) {
///             TableColumn("Base price") { purchase in
///                 Text(purchase.price, format: .currency(code: "USD"))
///             }
///             TableColumn("With 15% tip") { purchase in
///                 Text(purchase.price * 1.15, format: .currency(code: "USD"))
///             }
///             TableColumn("With 20% tip") { purchase in
///                 Text(purchase.price * 1.2, format: .currency(code: "USD"))
///             }
///         } rows: {
///             TableRow(Purchase(price: 20))
///             TableRow(Purchase(price: 50))
///             TableRow(Purchase(price: 75))
///         }
///         .introspect(.table, on: .iOS(.v16, .v17, .v18)) {
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
///     struct Purchase: Identifiable {
///         let id = UUID()
///         let price: Decimal
///     }
///
///     var body: some View {
///         Table(of: Purchase.self) {
///             TableColumn("Base price") { purchase in
///                 Text(purchase.price, format: .currency(code: "USD"))
///             }
///             TableColumn("With 15% tip") { purchase in
///                 Text(purchase.price * 1.15, format: .currency(code: "USD"))
///             }
///             TableColumn("With 20% tip") { purchase in
///                 Text(purchase.price * 1.2, format: .currency(code: "USD"))
///             }
///         } rows: {
///             TableRow(Purchase(price: 20))
///             TableRow(Purchase(price: 50))
///             TableRow(Purchase(price: 75))
///         }
///         .introspect(.table, on: .macOS(.v12, .v13, .v14, .v15)) {
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
///     struct Purchase: Identifiable {
///         let id = UUID()
///         let price: Decimal
///     }
///
///     var body: some View {
///         Table(of: Purchase.self) {
///             TableColumn("Base price") { purchase in
///                 Text(purchase.price, format: .currency(code: "USD"))
///             }
///             TableColumn("With 15% tip") { purchase in
///                 Text(purchase.price * 1.15, format: .currency(code: "USD"))
///             }
///             TableColumn("With 20% tip") { purchase in
///                 Text(purchase.price * 1.2, format: .currency(code: "USD"))
///             }
///         } rows: {
///             TableRow(Purchase(price: 20))
///             TableRow(Purchase(price: 50))
///             TableRow(Purchase(price: 75))
///         }
///         .introspect(.table, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UICollectionView
///         }
///     }
/// }
/// ```
public struct TableType: IntrospectableViewType {}

#if !os(tvOS)
extension IntrospectableViewType where Self == TableType {
    public static var table: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<TableType, UICollectionView> {
    @available(*, unavailable, message: "Table isn't available on iOS 13")
    public static let v13 = Self(for: .v13)
    @available(*, unavailable, message: "Table isn't available on iOS 14")
    public static let v14 = Self(for: .v14)
    @available(*, unavailable, message: "Table isn't available on iOS 15")
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<TableType, UICollectionView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<TableType, NSTableView> {
    @available(*, unavailable, message: "Table isn't available on macOS 10.15")
    public static let v10_15 = Self(for: .v10_15)
    @available(*, unavailable, message: "Table isn't available on macOS 11")
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
