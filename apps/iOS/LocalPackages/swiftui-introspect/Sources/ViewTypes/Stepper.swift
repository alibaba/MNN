#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Stepper` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var selection = 5
///
///     var body: some View {
///         Stepper("Select a number", value: $selection, in: 0...10)
///             .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UIStepper
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
///     @State var selection = 5
///
///     var body: some View {
///         Stepper("Select a number", value: $selection, in: 0...10)
///             .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSStepper
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// Not available.
public struct StepperType: IntrospectableViewType {}

#if !os(tvOS) && !os(visionOS)
extension IntrospectableViewType where Self == StepperType {
    public static var stepper: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<StepperType, UIStepper> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}
#elseif canImport(AppKit)
extension macOSViewVersion<StepperType, NSStepper> {
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
