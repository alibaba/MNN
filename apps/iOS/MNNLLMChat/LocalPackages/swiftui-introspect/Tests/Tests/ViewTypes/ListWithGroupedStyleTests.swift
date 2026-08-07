#if !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ListWithGroupedStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformListWithGroupedStyle = UIScrollView // covers both UITableView and UICollectionView
    #endif

    func testListWithGroupedStyle() {
        XCTAssertViewIntrospection(of: PlatformListWithGroupedStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                .listStyle(.grouped)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.list(style: .grouped), on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy0($0) }
                .introspect(.list(style: .grouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #endif

                List {
                    Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.list(style: .grouped), on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { spy1($0) }
                    .introspect(.list(style: .grouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    #endif
                }
                .listStyle(.grouped)
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
#endif
