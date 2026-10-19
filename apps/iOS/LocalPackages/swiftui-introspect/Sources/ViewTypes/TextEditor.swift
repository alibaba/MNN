#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `TextEditor` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         TextEditor(text: $text)
///             .introspect(.textEditor, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UITextView
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
///     @State var text = "Lorem ipsum"
///
///     var body: some View {
///         TextEditor(text: $text)
///             .introspect(.textEditor, on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSTextView
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
///         TextEditor(text: $text)
///             .introspect(.textEditor, on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UITextView
///             }
///     }
/// }
/// ```
public struct TextEditorType: IntrospectableViewType {}

#if !os(tvOS)
extension IntrospectableViewType where Self == TextEditorType {
    public static var textEditor: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<TextEditorType, UITextView> {
    @available(*, unavailable, message: "TextEditor isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<TextEditorType, UITextView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<TextEditorType, NSTextView> {
    @available(*, unavailable, message: "TextEditor isn't available on macOS 10.15")
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
