#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `List` type in SwiftUI, with `.bordered` style.
///
/// ### iOS
///
/// Not available.
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
///         .listStyle(.bordered)
///         .introspect(.list(style: .bordered), on: .macOS(.v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSTableView
///         }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct ListWithBorderedStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case bordered
    }
}

#if !os(iOS) && !os(tvOS) && !os(visionOS)
extension IntrospectableViewType where Self == ListWithBorderedStyleType {
    public static func list(style: Self.Style) -> Self { .init() }
}

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
extension macOSViewVersion<ListWithBorderedStyleType, NSTableView> {
    @available(*, unavailable, message: ".listStyle(.insetGrouped) isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    @available(*, unavailable, message: ".listStyle(.insetGrouped) isn't available on macOS 11")
    public static let v11 = Self.unavailable()
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
