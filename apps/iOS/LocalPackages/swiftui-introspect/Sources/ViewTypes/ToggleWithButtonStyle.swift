#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Toggle` type in SwiftUI, with `.button` style.
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
///     @State var isOn = false
///
///     var body: some View {
///         Toggle("Toggle", isOn: $isOn)
///             .toggleStyle(.button)
///             .introspect(.toggle(style: .button), on: .macOS(.v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSButton
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct ToggleWithButtonStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case button
    }
}

#if !os(iOS) && !os(tvOS) && !os(visionOS)
extension IntrospectableViewType where Self == ToggleWithButtonStyleType {
    public static func toggle(style: Self.Style) -> Self { .init() }
}

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
extension macOSViewVersion<ToggleWithButtonStyleType, NSButton> {
    @available(*, unavailable, message: ".toggleStyle(.button) isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    @available(*, unavailable, message: ".toggleStyle(.button) isn't available on macOS 11")
    public static let v11 = Self.unavailable()
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
