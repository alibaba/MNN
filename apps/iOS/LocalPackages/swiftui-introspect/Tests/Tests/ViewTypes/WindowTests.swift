import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class WindowTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformWindow = UIWindow
    #elseif canImport(AppKit)
    typealias PlatformWindow = NSWindow
    #endif

    func testWindow() {
        XCTAssertViewIntrospection(of: PlatformWindow.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Image(systemName: "scribble")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.window, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.window, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                Text("Text")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.window, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.window, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
            }
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(.window, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
            #elseif os(macOS)
            .introspect(.window, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
            #endif
        } extraAssertions: {
            XCTAssertIdentical($0[safe: 0], $0[safe: 1])
            XCTAssertIdentical($0[safe: 0], $0[safe: 2])
            XCTAssertIdentical($0[safe: 1], $0[safe: 2])
        }
    }
}
