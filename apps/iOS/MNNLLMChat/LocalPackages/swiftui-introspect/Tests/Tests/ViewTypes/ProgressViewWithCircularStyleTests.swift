import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ProgressViewWithCircularStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformProgressViewWithCircularStyle = UIActivityIndicatorView
    #elseif canImport(AppKit)
    typealias PlatformProgressViewWithCircularStyle = NSProgressIndicator
    #endif

    func testProgressViewWithCircularStyle() throws {
        guard #available(iOS 14, tvOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformProgressViewWithCircularStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                ProgressView(value: 0.25)
                    .progressViewStyle(.circular)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .circular), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .circular), on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                ProgressView(value: 0.5)
                    .progressViewStyle(.circular)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .circular), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .circular), on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                ProgressView(value: 0.75)
                    .progressViewStyle(.circular)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .circular), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .circular), on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(AppKit) && !targetEnvironment(macCatalyst)
            XCTAssertEqual($0[safe: 0]?.doubleValue, 0.25)
            XCTAssertEqual($0[safe: 1]?.doubleValue, 0.5)
            XCTAssertEqual($0[safe: 2]?.doubleValue, 0.75)
            #endif
        }
    }
}
