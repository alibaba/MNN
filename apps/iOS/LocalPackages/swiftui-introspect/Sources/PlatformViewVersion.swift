#if !os(watchOS)
import SwiftUI

@MainActor
public struct PlatformViewVersionPredicate<SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity> {
    let selector: IntrospectionSelector<PlatformSpecificEntity>?

    private init<Version: PlatformVersion>(
        _ versions: [PlatformViewVersion<Version, SwiftUIViewType, PlatformSpecificEntity>],
        matches: (PlatformViewVersion<Version, SwiftUIViewType, PlatformSpecificEntity>) -> Bool
    ) {
        if let matchingVersion = versions.first(where: matches) {
            self.selector = matchingVersion.selector ?? .default
        } else {
            self.selector = nil
        }
    }

    public static func iOS(_ versions: (iOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>)...) -> Self {
        Self(versions, matches: \.isCurrent)
    }

    @_spi(Advanced)
    public static func iOS(_ versions: PartialRangeFrom<iOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>>) -> Self {
        Self([versions.lowerBound], matches: \.isCurrentOrPast)
    }

    public static func tvOS(_ versions: (tvOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>)...) -> Self {
        Self(versions, matches: \.isCurrent)
    }

    @_spi(Advanced)
    public static func tvOS(_ versions: PartialRangeFrom<tvOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>>) -> Self {
        Self([versions.lowerBound], matches: \.isCurrentOrPast)
    }

    public static func macOS(_ versions: (macOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>)...) -> Self {
        Self(versions, matches: \.isCurrent)
    }

    @_spi(Advanced)
    public static func macOS(_ versions: PartialRangeFrom<macOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>>) -> Self {
        Self([versions.lowerBound], matches: \.isCurrentOrPast)
    }

    public static func visionOS(_ versions: (visionOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>)...) -> Self {
        Self(versions, matches: \.isCurrent)
    }

    @_spi(Advanced)
    public static func visionOS(_ versions: PartialRangeFrom<visionOSViewVersion<SwiftUIViewType, PlatformSpecificEntity>>) -> Self {
        Self([versions.lowerBound], matches: \.isCurrentOrPast)
    }
}

public typealias iOSViewVersion<SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity> =
    PlatformViewVersion<iOSVersion, SwiftUIViewType, PlatformSpecificEntity>
public typealias tvOSViewVersion<SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity> =
    PlatformViewVersion<tvOSVersion, SwiftUIViewType, PlatformSpecificEntity>
public typealias macOSViewVersion<SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity> =
    PlatformViewVersion<macOSVersion, SwiftUIViewType, PlatformSpecificEntity>
public typealias visionOSViewVersion<SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity> =
    PlatformViewVersion<visionOSVersion, SwiftUIViewType, PlatformSpecificEntity>

@MainActor
public enum PlatformViewVersion<Version: PlatformVersion, SwiftUIViewType: IntrospectableViewType, PlatformSpecificEntity: PlatformEntity>: Sendable {
    @_spi(Internals) case available(Version, IntrospectionSelector<PlatformSpecificEntity>?)
    @_spi(Internals) case unavailable

    @_spi(Advanced) public init(for version: Version, selector: IntrospectionSelector<PlatformSpecificEntity>? = nil) {
        self = .available(version, selector)
    }

    @_spi(Advanced) public static func unavailable(file: StaticString = #file, line: UInt = #line) -> Self {
        let filePath = file.withUTF8Buffer { String(decoding: $0, as: UTF8.self) }
        let fileName = URL(fileURLWithPath: filePath).lastPathComponent
        print(
            """
            If you're seeing this, someone forgot to mark \(fileName):\(line) as unavailable.

            This won't have any effect, but it should be disallowed altogether.

            Please report it upstream so we can properly fix it by using the following link:

            https://github.com/siteline/swiftui-introspect/issues/new?title=`\(fileName):\(line)`+should+be+marked+unavailable
            """
        )
        return .unavailable
    }

    private var version: Version? {
        if case .available(let version, _) = self {
            return version
        } else {
            return nil
        }
    }

    @MainActor
    fileprivate var selector: IntrospectionSelector<PlatformSpecificEntity>? {
        if case .available(_, let selector) = self {
            return selector
        } else {
            return nil
        }
    }

    fileprivate var isCurrent: Bool {
        version?.isCurrent ?? false
    }

    fileprivate var isCurrentOrPast: Bool {
        version?.isCurrentOrPast ?? false
    }
}

// This conformance isn't meant to be used directly by the user,
// it's only to satisfy requirements for forming ranges (e.g. `.v15...`).
extension PlatformViewVersion: Comparable {
    nonisolated public static func == (lhs: Self, rhs: Self) -> Bool {
        true
    }

    nonisolated public static func < (lhs: Self, rhs: Self) -> Bool {
        true
    }
}
#endif
