import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ListCellTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformListCell = UIView // covers both UITableViewCell and UICollectionViewCell
    #elseif canImport(AppKit)
    typealias PlatformListCell = NSTableCellView
    #endif

    func testListCell() {
        XCTAssertViewIntrospection(of: PlatformListCell.self) { spies in
            let spy = spies[0]

            List {
                Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.listCell, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy($0) }
                    .introspect(.listCell, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy($0) }
                    #elseif os(macOS)
                    .introspect(.listCell, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { spy($0) }
                    #endif
            }
        }
    }

    func testMaskedListCell() {
        XCTAssertViewIntrospection(of: PlatformListCell.self) { spies in
            let spy = spies[0]

            List {
                Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.listCell, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy($0) }
                    .introspect(.listCell, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy($0) }
                    #elseif os(macOS)
                    .introspect(.listCell, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { spy($0) }
                    #endif
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 20.0))
                    .cornerRadius(2.0)
            }
        }
    }
}
