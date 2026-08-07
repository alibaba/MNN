@_spi(Advanced)
@propertyWrapper
public final class Weak<T: AnyObject> {
    private weak var _wrappedValue: T?

    public var wrappedValue: T? {
        get { _wrappedValue }
        set { _wrappedValue = newValue }
    }

    public init(wrappedValue: T? = nil) {
        self._wrappedValue = wrappedValue
    }
}
