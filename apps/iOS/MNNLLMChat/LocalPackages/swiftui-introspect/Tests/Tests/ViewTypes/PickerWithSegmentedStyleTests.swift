import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class PickerWithSegmentedStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformPickerWithSegmentedStyle = UISegmentedControl
    #elseif canImport(AppKit)
    typealias PlatformPickerWithSegmentedStyle = NSSegmentedControl
    #endif

    func testPickerWithSegmentedStyle() {
        XCTAssertViewIntrospection(of: PlatformPickerWithSegmentedStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                }
                .pickerStyle(.segmented)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.picker(style: .segmented), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                #elseif os(macOS)
                .introspect(.picker(style: .segmented), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                }
                .pickerStyle(.segmented)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.picker(style: .segmented), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                #elseif os(macOS)
                .introspect(.picker(style: .segmented), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                    Text("3").tag("3")
                }
                .pickerStyle(.segmented)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.picker(style: .segmented), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                #elseif os(macOS)
                .introspect(.picker(style: .segmented), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.numberOfSegments, 1)
            XCTAssertEqual($0[safe: 1]?.numberOfSegments, 2)
            XCTAssertEqual($0[safe: 2]?.numberOfSegments, 3)
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.segmentCount, 1)
            XCTAssertEqual($0[safe: 1]?.segmentCount, 2)
            XCTAssertEqual($0[safe: 2]?.segmentCount, 3)
            #endif
        }
    }
}
