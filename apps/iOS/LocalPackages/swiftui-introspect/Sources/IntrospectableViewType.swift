#if !os(watchOS)
@MainActor
public protocol IntrospectableViewType {
    /// The scope of introspection for this particular view type, i.e. where introspect
    /// should look to find the desired target view relative to the applied
    /// `.introspect(...)` modifier.
    ///
    /// While the scope can be overridden by the user in their `.introspect(...)` call,
    /// most of the time it's preferable to defer to the view type's own scope,
    /// as it guarantees introspection is working as intended by the vendor.
    ///
    /// Defaults to `.receiver` if left unimplemented, which is a sensible one in
    /// most cases if you're looking to implement your own view type.
    var scope: IntrospectionScope { get }
}

extension IntrospectableViewType {
    public var scope: IntrospectionScope { .receiver }
}
#endif
