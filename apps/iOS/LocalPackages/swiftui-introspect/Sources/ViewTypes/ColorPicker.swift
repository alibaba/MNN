#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `ColorPicker` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var color = Color.red
///
///     var body: some View {
///         ColorPicker("Pick a color", selection: $color)
///             .introspect(.colorPicker, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UIColorPicker
///             }
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
///     @State var color = Color.red
///
///     var body: some View {
///         ColorPicker("Pick a color", selection: $color)
///             .introspect(.colorPicker, on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSColorPicker
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var color = Color.red
///
///     var body: some View {
///         ColorPicker("Pick a color", selection: $color)
///             .introspect(.colorPicker, on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UIColorPicker
///             }
///     }
/// }
/// ```
public struct ColorPickerType: IntrospectableViewType {}

#if !os(tvOS)
extension IntrospectableViewType where Self == ColorPickerType {
    public static var colorPicker: Self { .init() }
}

#if canImport(UIKit)
@available(iOS 14, *)
extension iOSViewVersion<ColorPickerType, UIColorWell> {
    @available(*, unavailable, message: "ColorPicker isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

@available(iOS 14, *)
extension visionOSViewVersion<ColorPickerType, UIColorWell> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
@available(macOS 11, *)
extension macOSViewVersion<ColorPickerType, NSColorWell> {
    @available(*, unavailable, message: "ColorPicker isn't available on macOS 10.15")
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
