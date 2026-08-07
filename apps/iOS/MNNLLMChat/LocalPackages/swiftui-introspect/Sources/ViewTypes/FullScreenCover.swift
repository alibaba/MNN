#if !os(watchOS)
import SwiftUI

/// An abstract representation of `.fullScreenCover` in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var isPresented = false
///
///     var body: some View {
///         Button("Present", action: { isPresented = true })
///             .fullScreenCover(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.fullScreenCover, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///                         print(type(of: $0)) // UIPresentationController
///                     }
///             }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     @State var isPresented = false
///
///     var body: some View {
///         Button("Present", action: { isPresented = true })
///             .fullScreenCover(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.fullScreenCover, on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///                         print(type(of: $0)) // UIPresentationController
///                     }
///             }
///     }
/// }
/// ```
///
/// ### macOS
///
/// Not available.
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var isPresented = false
///
///     var body: some View {
///         Button("Present", action: { isPresented = true })
///             .fullScreenCover(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.fullScreenCover, on: .visionOS(.v1, .v2)) {
///                         print(type(of: $0)) // UIPresentationController
///                     }
///             }
///     }
/// }
/// ```
public struct FullScreenCoverType: IntrospectableViewType {
    public var scope: IntrospectionScope { .ancestor }
}

#if !os(macOS)
extension IntrospectableViewType where Self == FullScreenCoverType {
    public static var fullScreenCover: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<FullScreenCoverType, UIPresentationController> {
    @available(*, unavailable, message: ".fullScreenCover isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UIPresentationController> {
        .from(UIViewController.self, selector: { $0.presentationController })
    }
}

extension tvOSViewVersion<FullScreenCoverType, UIPresentationController> {
    @available(*, unavailable, message: ".fullScreenCover isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UIPresentationController> {
        .from(UIViewController.self, selector: { $0.presentationController })
    }
}

extension visionOSViewVersion<FullScreenCoverType, UIPresentationController> {
    public static let v1 = Self(for: .v1, selector: selector)
    public static let v2 = Self(for: .v2, selector: selector)

    private static var selector: IntrospectionSelector<UIPresentationController> {
        .from(UIViewController.self, selector: { $0.presentationController })
    }
}
#endif
#endif
#endif
