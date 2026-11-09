#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `DatePicker` type in SwiftUI, with `.graphical` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.graphical)
///             .introspect(.datePicker(style: .graphical), on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
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
/// ```swift
/// struct ContentView: View {
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.graphical)
///             .introspect(.datePicker(style: .graphical), on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSDatePicker
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var date = Date()
///
///     var body: some View {
///         DatePicker("Pick a date", selection: $date)
///             .datePickerStyle(.graphical)
///             .introspect(.datePicker(style: .graphical), on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UIDatePicker
///             }
///     }
/// }
/// ```
public struct DatePickerWithGraphicalStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case graphical
    }
}

#if !os(tvOS)
extension IntrospectableViewType where Self == DatePickerWithGraphicalStyleType {
    public static func datePicker(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<DatePickerWithGraphicalStyleType, UIDatePicker> {
    @available(*, unavailable, message: ".datePickerStyle(.graphical) isn't available on iOS 13")
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<DatePickerWithGraphicalStyleType, UIDatePicker> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit) && !targetEnvironment(macCatalyst)
extension macOSViewVersion<DatePickerWithGraphicalStyleType, NSDatePicker> {
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
