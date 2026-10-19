import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 16, tvOS 16, macOS 13, *)
@MainActor
final class FormWithGroupedStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformFormWithGroupedStyle = UIScrollView // covers both UITableView and UICollectionView
    #elseif canImport(AppKit)
    typealias PlatformFormWithGroupedStyle = NSScrollView
    #endif

    func testFormWithGroupedStyle() throws {
        guard #available(iOS 16, tvOS 16, macOS 13, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformFormWithGroupedStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                Form {
                    Text("Item 1")
                }
                .formStyle(.grouped)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.form(style: .grouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { spy0($0) }
                .introspect(.form(style: .grouped), on: .tvOS(.v16, .v17, .v18)) { spy0($0) }
                #elseif os(macOS)
                .introspect(.form(style: .grouped), on: .macOS(.v13, .v14)) { spy0($0) }
                #endif

                Form {
                    Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.form(style: .grouped), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { spy1($0) }
                    .introspect(.form(style: .grouped), on: .tvOS(.v16, .v17, .v18), scope: .ancestor) { spy1($0) }
                    #elseif os(macOS)
                    .introspect(.form(style: .grouped), on: .macOS(.v13, .v14), scope: .ancestor) { spy1($0) }
                    #endif
                }
                .formStyle(.grouped)
            }
        } extraAssertions: {
            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
