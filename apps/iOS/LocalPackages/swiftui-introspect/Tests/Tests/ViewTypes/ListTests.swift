import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ListTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformList = UIScrollView // covers both UITableView and UICollectionView
    #elseif canImport(AppKit)
    typealias PlatformList = NSTableView
    #endif

    func testList() {
        XCTAssertViewIntrospection(of: PlatformList.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy0($0) }
                .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #elseif os(macOS)
                .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { spy0($0) }
                #endif

                List {
                    Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { spy1($0) }
                    .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    #elseif os(macOS)
                    .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor) { spy1($0) }
                    #endif
                }
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }

    #if !os(macOS)
    func testNestedList() {
        XCTAssertViewIntrospection(of: PlatformList.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            List {
                Text("Item 1")

                List {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy1($0) }
                .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy1($0) }
                #endif
            }
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy0($0) }
            .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
            #endif
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
    #endif

    func testMaskedList() {
        XCTAssertViewIntrospection(of: PlatformList.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                List {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy0($0) }
                .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #elseif os(macOS)
                .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { spy0($0) }
                #endif
                .clipped()
                .clipShape(RoundedRectangle(cornerRadius: 20.0))
                .cornerRadius(2.0)

                List {
                    Text("Item 1")
                        #if os(iOS) || os(tvOS) || os(visionOS)
                        .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { spy1($0) }
                        .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                        #elseif os(macOS)
                        .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor) { spy1($0) }
                        #endif
                }
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
