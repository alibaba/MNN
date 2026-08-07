#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Picker` type in SwiftUI, with `.menu` style.
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
///     @State var selection = "1"
///
///     var body: some View {
///         Picker("Pick a number", selection: $selection) {
///             Text("1").tag("1")
///             Text("2").tag("2")
///             Text("3").tag("3")
///         }
///         .pickerStyle(.menu)
///         .introspect(.picker(style: .menu), on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSPopUpButton
///         }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct PickerWithMenuStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case menu
    }
}

#if !os(iOS) && !os(tvOS) && !os(visionOS)
extension IntrospectableViewType where Self == PickerWithMenuStyleType {
    public static func picker(style: Self.Style) -> Self { .init() }
}

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
extension macOSViewVersion<PickerWithMenuStyleType, NSPopUpButton> {
    @available(*, unavailable, message: ".pickerStyle(.menu) isn't available on macOS 10.15")
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
