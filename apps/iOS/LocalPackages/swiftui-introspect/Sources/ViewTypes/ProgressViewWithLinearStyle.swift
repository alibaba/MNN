#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `ProgressView` type in SwiftUI, with `.linear` style.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         ProgressView(value: 0.5)
///             .progressViewStyle(.linear)
///             .introspect(.progressView(style: .linear), on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UIProgressView
///             }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         ProgressView(value: 0.5)
///             .progressViewStyle(.linear)
///             .introspect(.progressView(style: .linear), on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // UIProgressView
///             }
///     }
/// }
/// ```
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         ProgressView(value: 0.5)
///             .progressViewStyle(.linear)
///             .introspect(.progressView(style: .linear), on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // NSProgressIndicator
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         ProgressView(value: 0.5)
///             .progressViewStyle(.linear)
///             .introspect(.progressView(style: .linear), on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // UIProgressView
///             }
///     }
/// }
/// ```
public struct ProgressViewWithLinearStyleType: IntrospectableViewType {
    public enum Style: Sendable {
        case linear
    }
}

extension IntrospectableViewType where Self == ProgressViewWithLinearStyleType {
    public static func progressView(style: Self.Style) -> Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<ProgressViewWithLinearStyleType, UIProgressView> {
    @available(*, unavailable, message: ".progressViewStyle(.linear) isn't available on iOS 13")
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<ProgressViewWithLinearStyleType, UIProgressView> {
    @available(*, unavailable, message: ".progressViewStyle(.linear) isn't available on tvOS 13")
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension visionOSViewVersion<ProgressViewWithLinearStyleType, UIProgressView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#elseif canImport(AppKit)
extension macOSViewVersion<ProgressViewWithLinearStyleType, NSProgressIndicator> {
    @available(*, unavailable, message: ".progressViewStyle(.linear) isn't available on macOS 10.15")
    public static let v10_15 = Self(for: .v10_15)
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}
#endif
#endif
