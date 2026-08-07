#if !os(watchOS)
@_spi(Advanced)

@MainActor
public struct IntrospectionSelector<Target: PlatformEntity> {
    @_spi(Advanced)
    public static var `default`: Self { .from(Target.self, selector: { $0 }) }

    @_spi(Advanced)
    public static func from<Entry: PlatformEntity>(_ entryType: Entry.Type, selector: @MainActor @escaping (Entry) -> Target?) -> Self {
        .init(
            receiverSelector: { controller in
                controller.as(Entry.Base.self)?.receiver(ofType: Entry.self).flatMap(selector)
            },
            ancestorSelector: { controller in
                controller.as(Entry.Base.self)?.ancestor(ofType: Entry.self).flatMap(selector)
            }
        )
    }

    private var receiverSelector: @MainActor (IntrospectionPlatformViewController) -> Target?
    private var ancestorSelector: @MainActor (IntrospectionPlatformViewController) -> Target?

    private init(
        receiverSelector: @MainActor @escaping (IntrospectionPlatformViewController) -> Target?,
        ancestorSelector: @MainActor @escaping (IntrospectionPlatformViewController) -> Target?
    ) {
        self.receiverSelector = receiverSelector
        self.ancestorSelector = ancestorSelector
    }

    @_spi(Advanced)
    public func withReceiverSelector(_ selector: @MainActor @escaping (PlatformViewController) -> Target?) -> Self {
        var copy = self
        copy.receiverSelector = selector
        return copy
    }

    @_spi(Advanced)
    public func withAncestorSelector(_ selector: @MainActor @escaping (PlatformViewController) -> Target?) -> Self {
        var copy = self
        copy.ancestorSelector = selector
        return copy
    }

    func callAsFunction(_ controller: IntrospectionPlatformViewController, _ scope: IntrospectionScope) -> Target? {
        if
            scope.contains(.receiver),
            let target = receiverSelector(controller)
        {
            return target
        }
        if
            scope.contains(.ancestor),
            let target = ancestorSelector(controller)
        {
            return target
        }
        return nil
    }
}

extension PlatformViewController {
    func `as`<Base: PlatformEntity>(_ baseType: Base.Type) -> (any PlatformEntity)? {
        if Base.self == PlatformView.self {
            #if canImport(UIKit)
            return viewIfLoaded
            #elseif canImport(AppKit)
            return isViewLoaded ? view : nil
            #endif
        } else if Base.self == PlatformViewController.self {
            return self
        }
        return nil
    }
}
#endif
