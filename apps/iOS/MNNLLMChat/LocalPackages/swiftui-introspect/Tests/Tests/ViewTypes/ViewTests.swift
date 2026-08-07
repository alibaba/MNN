import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ViewTests: XCTestCase {
    func testView() {
        XCTAssertViewIntrospection(of: PlatformView.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack(spacing: 10) {
                Image(systemName: "scribble").resizable().frame(height: 30)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.view, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                Text("Text").frame(height: 40)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.view, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
            }
            .padding(10)
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(.view, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
            #elseif os(macOS)
            .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
            #endif
        } extraAssertions: {
            XCTAssertEqual($0[safe: 0]?.frame.height, 30)
            XCTAssertEqual($0[safe: 1]?.frame.height, 40)
            XCTAssertEqual($0[safe: 2]?.frame.height, 100)
        }
    }
}
