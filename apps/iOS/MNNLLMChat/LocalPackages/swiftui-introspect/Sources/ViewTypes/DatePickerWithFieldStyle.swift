#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `DatePicker` type in SwiftUI, with `.field` style.
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
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.field)
///             .introspect(.datePicker(style: .field), on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSDatePicker
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct DatePickerWithFieldStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case field
    }
}

#if !os(iOS) && !os(tvOS) && !os(visionOS)
extension IntrospectableViewType where Self == DatePickerWithFieldStyleType {
    public static func datePicker(style: Self.Style) -> Self { .init() }
}

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
extension macOSViewVersion<DatePickerWithFieldStyleType, NSDatePicker> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
#endif
