#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, macOS 11, *)
@MainActor
final class ListWithInsetStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformListWithInsetStyle = UIScrollView // covers both UITableView and UICollectionView
    #elseif canImport(AppKit)
    typealias PlatformListWithInsetStyle = NSTableView
    #endif

    func testListWithInsetStyle() throws {
        guard #available(iOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformListWithInsetStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                .listStyle(.inset)
                #if os(iOS) || os(visionOS)
                .introspect(.list(style: .inset), on: .iOS(.v14, .v15)) { spy0($0) }
                .introspect(.list(style: .inset), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #elseif os(macOS)
                .introspect(.list(style: .inset), on: .macOS(.v11, .v12, .v13, .v14)) { spy0($0) }
                #endif

                List {
                    Text("Item 1")
                    #if os(iOS) || os(visionOS)
                    .introspect(.list(style: .inset), on: .iOS(.v14, .v15), scope: .ancestor) { spy1($0) }
                    .introspect(.list(style: .inset), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    #elseif os(macOS)
                    .introspect(.list(style: .inset), on: .macOS(.v11, .v12, .v13, .v14), scope: .ancestor) { spy1($0) }
                    #endif
                }
                .listStyle(.inset)
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
#endif
