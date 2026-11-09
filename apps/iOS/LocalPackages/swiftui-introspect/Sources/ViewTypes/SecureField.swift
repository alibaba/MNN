#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `SecureField` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         SecureField("Secure Field", text: $text)
///             .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UISecureField
///             }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         SecureField("Secure Field", text: $text)
///             .introspect(.secureField, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UISecureField
///             }
///     }
/// }
/// ```
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         SecureField("Secure Field", text: $text)
///             .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSSecureField
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         SecureField("Secure Field", text: $text)
///             .introspect(.secureField, on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UISecureField
///             }
///     }
/// }
/// ```
public struct SecureFieldType: IntrospectableViewType {}

extension IntrospectableViewType where Self == SecureFieldType {
    public static var secureField: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<SecureFieldType, UITextField> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<SecureFieldType, UITextField> {
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<SecureFieldType, UITextField> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<SecureFieldType, NSTextField> {
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
