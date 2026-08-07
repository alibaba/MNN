#if !os(watchOS)
import SwiftUI

/// An abstract representation of the search field displayed via the `.searchable` modifier in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var searchTerm = ""
///
///     var body: some View {
///         NavigationView {
///             Text("Root")
///                 .searchable(text: $searchTerm)
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.searchField, on: .iOS(.v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UISearchBar
///         }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     @State var searchTerm = ""
///
///     var body: some View {
///         NavigationView {
///             Text("Root")
///                 .searchable(text: $searchTerm)
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.searchField, on: .tvOS(.v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // UISearchBar
///         }
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
///     @State var searchTerm = ""
///
///     var body: some View {
///         NavigationView {
///             Text("Root")
///                 .searchable(text: $searchTerm)
///         }
///         .navigationViewStyle(.stack)
///         .introspect(.searchField, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // UISearchBar
///         }
///     }
/// }
/// ```
public struct SearchFieldType: IntrospectableViewType {}

extension IntrospectableViewType where Self == SearchFieldType {
    public static var searchField: Self { .init() }
}

#if canImport(UIKit)
extension iOSViewVersion<SearchFieldType, UISearchBar> {
    @available(*, unavailable, message: ".searchable isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: ".searchable isn't available on iOS 14")
    public static let v14 = Self.unavailable()
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UISearchBar> {
        .from(UINavigationController.self) {
            $0.viewIfLoaded?.allDescendants.lazy.compactMap { $0 as? UISearchBar }.first
        }
    }
}

extension tvOSViewVersion<SearchFieldType, UISearchBar> {
    @available(*, unavailable, message: ".searchable isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    @available(*, unavailable, message: ".searchable isn't available on tvOS 14")
    public static let v14 = Self.unavailable()
    public static let v15 = Self(for: .v15, selector: selector)
    public static let v16 = Self(for: .v16, selector: selector)
    public static let v17 = Self(for: .v17, selector: selector)
    public static let v18 = Self(for: .v18, selector: selector)

    private static var selector: IntrospectionSelector<UISearchBar> {
        .from(UINavigationController.self) {
            $0.viewIfLoaded?.allDescendants.lazy.compactMap { $0 as? UISearchBar }.first
        }
    }
}

extension visionOSViewVersion<SearchFieldType, UISearchBar> {
    public static let v1 = Self(for: .v1, selector: selector)
    public static let v2 = Self(for: .v2, selector: selector)

    private static var selector: IntrospectionSelector<UISearchBar> {
        .from(UINavigationController.self) {
            $0.viewIfLoaded?.allDescendants.lazy.compactMap { $0 as? UISearchBar }.first
        }
    }
}
#endif
#endif
