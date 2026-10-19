#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(macOS 12, *)
@MainActor
final class ListWithBorderedStyleTests: XCTestCase {
    #if canImport(AppKit)
    typealias PlatformListWithBorderedStyle = NSTableView
    #endif

    func testListWithBorderedStyle() throws {
        guard #available(macOS 12, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformListWithBorderedStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                .listStyle(.bordered)
                #if os(macOS)
                .introspect(.list(style: .bordered), on: .macOS(.v12, .v13, .v14)) { spy0($0) }
                #endif

                List {
                    Text("Item 1")
                    #if os(macOS)
                    .introspect(.list(style: .bordered), on: .macOS(.v12, .v13, .v14), scope: .ancestor) { spy1($0) }
                    #endif
                }
                .listStyle(.bordered)
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
#endif
