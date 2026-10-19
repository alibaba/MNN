#if !os(tvOS) && !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, *)
@MainActor
final class ListWithInsetGroupedStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformListWithInsetGroupedStyle = UIScrollView // covers both UITableView and UICollectionView
    #endif

    func testListWithInsetGroupedStyle() throws {
        guard #available(iOS 14, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformListWithInsetGroupedStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                .listStyle(.insetGrouped)
                #if os(iOS) || os(visionOS)
                .introspect(.list(style: .insetGrouped), on: .iOS(.v14, .v15)) { spy0($0) }
                .introspect(.list(style: .insetGrouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #endif

                List {
                    Text("Item 1")
                    #if os(iOS) || os(visionOS)
                    .introspect(.list(style: .insetGrouped), on: .iOS(.v14, .v15), scope: .ancestor) { spy1($0) }
                    .introspect(.list(style: .insetGrouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    #endif
                }
                .listStyle(.insetGrouped)
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
#endif
