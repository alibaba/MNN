#if !os(watchOS)
import Foundation

@_spi(Internals)
public enum PlatformVersionCondition: Sendable {
    case past
    case current
    case future
}

public protocol PlatformVersion: Sendable {
    @_spi(Internals)
    var condition: PlatformVersionCondition? { get }
}

extension PlatformVersion {
    @_spi(Internals)
    public var isCurrent: Bool {
        condition == .current
    }

    @_spi(Internals)
    public var isCurrentOrPast: Bool {
        condition == .current || condition == .past
    }
  
    @_spi(Internals)
    public var condition: PlatformVersionCondition? { nil }
}

public struct iOSVersion: PlatformVersion {
    @_spi(Internals)
    public let condition: PlatformVersionCondition?

    @_spi(Internals)
    public init(condition: () -> PlatformVersionCondition?) {
        self.condition = condition()
    }
}

extension iOSVersion {
    public static let v13 = iOSVersion {
        #if os(iOS)
        if #available(iOS 14, *) {
            return .past
        }
        if #available(iOS 13, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v14 = iOSVersion {
        #if os(iOS)
        if #available(iOS 15, *) {
            return .past
        }
        if #available(iOS 14, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v15 = iOSVersion {
        #if os(iOS)
        if #available(iOS 16, *) {
            return .past
        }
        if #available(iOS 15, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v16 = iOSVersion {
        #if os(iOS)
        if #available(iOS 17, *) {
            return .past
        }
        if #available(iOS 16, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v17 = iOSVersion {
        #if os(iOS)
        if #available(iOS 18, *) {
            return .past
        }
        if #available(iOS 17, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v18 = iOSVersion {
        #if os(iOS)
        if #available(iOS 19, *) {
            return .past
        }
        if #available(iOS 18, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }
}

public struct tvOSVersion: PlatformVersion {
    @_spi(Internals)
    public let condition: PlatformVersionCondition?

    @_spi(Internals)
    public init(condition: () -> PlatformVersionCondition?) {
        self.condition = condition()
    }
}

extension tvOSVersion {
    public static let v13 = tvOSVersion {
        #if os(tvOS)
        if #available(tvOS 14, *) {
            return .past
        }
        if #available(tvOS 13, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v14 = tvOSVersion {
        #if os(tvOS)
        if #available(tvOS 15, *) {
            return .past
        }
        if #available(tvOS 14, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v15 = tvOSVersion {
        #if os(tvOS)
        if #available(tvOS 16, *) {
            return .past
        }
        if #available(tvOS 15, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v16 = tvOSVersion {
        #if os(tvOS)
        if #available(tvOS 17, *) {
            return .past
        }
        if #available(tvOS 16, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }
    
    public static let v17 = tvOSVersion {
#if os(tvOS)
        if #available(tvOS 18, *) {
            return .past
        }
        if #available(tvOS 17, *) {
            return .current
        }
        return .future
#else
        return nil
#endif
    }

    public static let v18 = tvOSVersion {
        #if os(tvOS)
        if #available(tvOS 19, *) {
            return .past
        }
        if #available(tvOS 18, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }
}

public struct macOSVersion: PlatformVersion {
    @_spi(Internals)
    public let condition: PlatformVersionCondition?

    @_spi(Internals)
    public init(condition: () -> PlatformVersionCondition?) {
        self.condition = condition()
    }
}

extension macOSVersion {
    public static let v10_15 = macOSVersion {
        #if os(macOS)
        if #available(macOS 11, *) {
            return .past
        }
        if #available(macOS 10.15, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v10_15_4 = macOSVersion {
        #if os(macOS)
        if #available(macOS 11, *) {
            return .past
        }
        if #available(macOS 10.15.4, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v11 = macOSVersion {
        #if os(macOS)
        if #available(macOS 12, *) {
            return .past
        }
        if #available(macOS 11, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v12 = macOSVersion {
        #if os(macOS)
        if #available(macOS 13, *) {
            return .past
        }
        if #available(macOS 12, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v13 = macOSVersion {
        #if os(macOS)
        if #available(macOS 14, *) {
            return .past
        }
        if #available(macOS 13, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v14 = macOSVersion {
        #if os(macOS)
        if #available(macOS 15, *) {
            return .past
        }
        if #available(macOS 14, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v15 = macOSVersion {
        #if os(macOS)
        if #available(macOS 16, *) {
            return .past
        }
        if #available(macOS 15, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }
}

public struct visionOSVersion: PlatformVersion {
    @_spi(Internals)
    public let condition: PlatformVersionCondition?

    @_spi(Internals)
    public init(condition: () -> PlatformVersionCondition?) {
        self.condition = condition()
    }
}

extension visionOSVersion {
    public static let v1 = visionOSVersion {
        #if os(visionOS)
        if #available(visionOS 2, *) {
            return .past
        }
        if #available(visionOS 1, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }

    public static let v2 = visionOSVersion {
        #if os(visionOS)
        if #available(visionOS 3, *) {
            return .past
        }
        if #available(visionOS 2, *) {
            return .current
        }
        return .future
        #else
        return nil
        #endif
    }
}
#endif
