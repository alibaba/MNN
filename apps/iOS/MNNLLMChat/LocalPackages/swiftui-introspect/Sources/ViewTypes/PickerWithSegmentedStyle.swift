#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Picker` type in SwiftUI, with `.segmented` style.
///
/// ### iOS
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
///         .pickerStyle(.segmented)
///         .introspect(.picker(style: .segmented), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UISegmentedControl
///         }
///     }
/// }
/// ```
///
/// ### tvOS
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
///         .pickerStyle(.segmented)
///         .introspect(.picker(style: .segmented), on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UISegmentedControl
///         }
///     }
/// }
/// ```
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
///         .pickerStyle(.segmented)
///         .introspect(.picker(style: .segmented), on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///             print(type(of: $0)) // NSSegmentedControl
///         }
///     }
/// }
/// ```
///
/// ### visionOS
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
///         .pickerStyle(.segmented)
///         .introspect(.picker(style: .segmented), on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UISegmentedControl
///         }
///     }
/// }
/// ```
public struct PickerWithSegmentedStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case segmented
    }
}

extension IntrospectableViewType where Self == PickerWithSegmentedStyleType {
    public static func picker(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<PickerWithSegmentedStyleType, UISegmentedControl> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<PickerWithSegmentedStyleType, UISegmentedControl> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<PickerWithSegmentedStyleType, UISegmentedControl> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<PickerWithSegmentedStyleType, NSSegmentedControl> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
