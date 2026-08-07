import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ScrollViewTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformScrollView = UIScrollView
    #elseif canImport(AppKit)
    typealias PlatformScrollView = NSScrollView
    #endif

    func testScrollView() {
        XCTAssertViewIntrospection(of: PlatformScrollView.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                ScrollView(showsIndicators: false) {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                #elseif os(macOS)
                .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                #endif

                ScrollView(showsIndicators: true) {
                    Text("Item 1")
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor, customize: spy1)
                    #elseif os(macOS)
                    .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor, customize: spy1)
                    #endif
                }
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.showsVerticalScrollIndicator, false)
            XCTAssertEqual($0[safe: 1]?.showsVerticalScrollIndicator, true)
            #elseif canImport(AppKit)
            // FIXME: these assertions don't pass on macOS 12, not sure why... maybe callback is too premature in relation to view lifecycle?
            if #available(macOS 13, *) {
                XCTAssert($0[safe: 0]?.verticalScroller == nil)
                XCTAssert($0[safe: 1]?.verticalScroller != nil)
            }
            #endif

            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }

    func testNestedScrollView() {
        XCTAssertViewIntrospection(of: PlatformScrollView.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            ScrollView(showsIndicators: true) {
                Text("Item 1")

                ScrollView(showsIndicators: false) {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                #elseif os(macOS)
                .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                #endif
            }
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
            #elseif os(macOS)
            .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
            #endif
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.showsVerticalScrollIndicator, true)
            XCTAssertEqual($0[safe: 1]?.showsVerticalScrollIndicator, false)
            #elseif canImport(AppKit)
            // FIXME: these assertions don't pass on macOS 12, not sure why... maybe callback is too premature in relation to view lifecycle?
            if #available(macOS 13, *) {
                XCTAssert($0[safe: 0]?.verticalScroller != nil)
                XCTAssert($0[safe: 1]?.verticalScroller == nil)
            }
            #endif

            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }

    func testMaskedScrollView() {
        XCTAssertViewIntrospection(of: PlatformScrollView.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]

            HStack {
                ScrollView(showsIndicators: false) {
                    Text("Item 1")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                #elseif os(macOS)
                .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                #endif
                .clipped()
                .clipShape(RoundedRectangle(cornerRadius: 20.0))
                .cornerRadius(2.0)

                ScrollView(showsIndicators: true) {
                    Text("Item 1")
                        #if os(iOS) || os(tvOS) || os(visionOS)
                        .introspect(.scrollView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor, customize: spy1)
                        #elseif os(macOS)
                        .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor, customize: spy1)
                        #endif
                }
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.showsVerticalScrollIndicator, false)
            XCTAssertEqual($0[safe: 1]?.showsVerticalScrollIndicator, true)
            #elseif canImport(AppKit)
            // FIXME: these assertions don't pass on macOS 12, not sure why... maybe callback is too premature in relation to view lifecycle?
            if #available(macOS 13, *) {
                XCTAssert($0[safe: 0]?.verticalScroller == nil)
                XCTAssert($0[safe: 1]?.verticalScroller != nil)
            }
            #endif

            XCTAssert($0[safe: 0] !== $0[safe: 1])
        }
    }
}
