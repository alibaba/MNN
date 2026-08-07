#if !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class FormTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformForm = UIScrollView // covers both UITableView and UICollectionView
    #elseif canImport(AppKit)
    typealias PlatformForm = NSScrollView
    #endif

    func testForm() throws {
        XCTAssertViewIntrospection(of: PlatformForm.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                Form {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.form, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { spy0($0) }
                .introspect(.form, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                #endif

                Form {
                    Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.form, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { spy1($0) }
                    .introspect(.form, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    #endif
                }
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
#endif
