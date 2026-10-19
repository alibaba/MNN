#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `DatePicker` type in SwiftUI, with `.wheel` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.wheel)
///             .introspect(.datePicker(style: .wheel), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UIDatePicker
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
/// Not available.
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.wheel)
///             .introspect(.datePicker(style: .wheel), on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UIDatePicker
///             }
///     }
/// }
/// ```
public struct DatePickerWithWheelStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case wheel
    }
}

#if !os(tvOS) && !os(macOS)
extension IntrospectableViewType where Self == DatePickerWithWheelStyleType {
    public static func datePicker(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<DatePickerWithWheelStyleType, UIDatePicker> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<DatePickerWithWheelStyleType, UIDatePicker> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#endif
#endif
#endif
