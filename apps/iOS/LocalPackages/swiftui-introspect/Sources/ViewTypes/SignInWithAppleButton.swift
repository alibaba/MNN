#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `SignInWithAppleButton` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         SignInWithAppleButton(.signIn) { request in
///             request.requestedScopes = [.fullName, .email]
///         } onCompletion: { result in
///             // do something with result
///         }
///         .introspect(.signInWithAppleButton, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // ASAuthorizationAppleIDButton
///         }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         SignInWithAppleButton(.signIn) { request in
///             request.requestedScopes = [.fullName, .email]
///         } onCompletion: { result in
///             // do something with result
///         }
///         .introspect(.signInWithAppleButton, on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///             print(type(of: $0)) // ASAuthorizationAppleIDButton
///         }
///     }
/// }
/// ```
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         SignInWithAppleButton(.signIn) { request in
///             request.requestedScopes = [.fullName, .email]
///         } onCompletion: { result in
///             // do something with result
///         }
///         .introspect(.signInWithAppleButton, on: .macOS(.v11, .v12, .v13, .v14),) {
///             print(type(of: $0)) // ASAuthorizationAppleIDButton
///         }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         SignInWithAppleButton(.signIn) { request in
///             request.requestedScopes = [.fullName, .email]
///         } onCompletion: { result in
///             // do something with result
///         }
///         .introspect(.signInWithAppleButton, on: .visionOS(.v1, .v2)) {
///             print(type(of: $0)) // ASAuthorizationAppleIDButton
///         }
///     }
/// }
/// ```
public struct SignInWithAppleButtonType: IntrospectableViewType {}

extension IntrospectableViewType where Self == SignInWithAppleButtonType {
    @available(
        *,
        unavailable,
        message: """
        Due to a mysterious bug on Apple's part that may cause a complete
        app hang, the unfortunate decision has been made to remove support
        for `SignInWithAppleButton` introspection.
        
        We apologize for this inconvenience.

        More details can be found at https://github.com/siteline/swiftui-introspect/issues/400
        """
    )
    public static var signInWithAppleButton: Self { .init() }
}
#endif
