import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ProgressViewWithLinearStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformProgressViewWithLinearStyle = UIProgressView
    #elseif canImport(AppKit)
    typealias PlatformProgressViewWithLinearStyle = NSProgressIndicator
    #endif

    func testProgressViewWithLinearStyle() throws {
        guard #available(iOS 14, tvOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformProgressViewWithLinearStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                ProgressView(value: 0.25)
                    .progressViewStyle(.linear)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .linear), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .linear), on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                ProgressView(value: 0.5)
                    .progressViewStyle(.linear)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .linear), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .linear), on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                ProgressView(value: 0.75)
                    .progressViewStyle(.linear)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.progressView(style: .linear), on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.progressView(style: .linear), on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.progress, 0.25)
            XCTAssertEqual($0[safe: 1]?.progress, 0.5)
            XCTAssertEqual($0[safe: 2]?.progress, 0.75)
            #elseif canImport(AppKit) && !targetEnvironment(macCatalyst)
            XCTAssertEqual($0[safe: 0]?.doubleValue, 0.25)
            XCTAssertEqual($0[safe: 1]?.doubleValue, 0.5)
            XCTAssertEqual($0[safe: 2]?.doubleValue, 0.75)
            #endif
        }
    }
}
