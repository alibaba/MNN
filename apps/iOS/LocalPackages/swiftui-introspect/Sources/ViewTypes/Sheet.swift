#if !os(watchOS)
import SwiftUI

/// An abstract representation of `.sheet` in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var isPresented = false
///
///     var body: some View {
///         Button("Present", action: { isPresented = true })
///             .sheet(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.sheet, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
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
///             .sheet(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.sheet, on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
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
///             .sheet(isPresented: $isPresented) {
///                 Button("Dismiss", action: { isPresented = false })
///                     .introspect(.sheet, on: .visionOS(.v1, .v2)) {
///                         print(type(of: $0)) // UISheetPresentationController
///                     }
///             }
///     }
/// }
/// ```
public struct SheetType: IntrospectableViewType {
    public var scope: IntrospectionScope { .ancestor }
}

#if !os(macOS)
extension IntrospectableViewType where Self == SheetType {
    public static var sheet: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<SheetType, UIPresentationController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UIPresentationController> {
        .from(UIViewController.self, selector: { $0.presentationController })
    }
}

#if !os(tvOS)
@available(iOS 15, *)
extension iOSViewVersion<SheetType, UISheetPresentationController> {
    @_disfavoredOverload
    public static let v15 = Self(for: .v15, selector: selector)
    @_disfavoredOverload
    public static let v16 = Self(for: .v16, selector: selector)
    @_disfavoredOverload
    public static let v17 = Self(for: .v17, selector: selector)
    @_disfavoredOverload
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UISheetPresentationController> {
        .from(UIViewController.self, selector: { $0.sheetPresentationController })
    }
}

@available(iOS 15, *)
extension visionOSViewVersion<SheetType, UISheetPresentationController> {
    public static let v1 = Self(for: .v1, selector: selector)
    public static let v2 = Self(for: .v2, selector: selector)

    private static var selector: IntrospectionSelector<UISheetPresentationController> {
        .from(UIViewController.self, selector: { $0.sheetPresentationController })
    }
}
#endif

extension tvOSViewVersion<SheetType, UIPresentationController> {
    public static let v13 = Self(for: .v13, selector: selector)
    public static let v14 = Self(for: .v14, selector: selector)
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UIPresentationController> {
        .from(UIViewController.self, selector: { $0.presentationController })
    }
}
#endif
#endif
#endif
