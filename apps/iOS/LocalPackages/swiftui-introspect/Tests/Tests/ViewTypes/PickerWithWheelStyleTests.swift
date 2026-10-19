#if !os(tvOS) && !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class PickerWithWheelStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformPickerWithWheelStyle = UIPickerView
    #endif

    func testPickerWithWheelStyle() {
        XCTAssertViewIntrospection(of: PlatformPickerWithWheelStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                }
                .pickerStyle(.wheel)
                #if os(iOS) || os(visionOS)
                .introspect(.picker(style: .wheel), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                }
                .pickerStyle(.wheel)
                #if os(iOS) || os(visionOS)
                .introspect(.picker(style: .wheel), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                    Text("3").tag("3")
                }
                .pickerStyle(.wheel)
                #if os(iOS) || os(visionOS)
                .introspect(.picker(style: .wheel), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.numberOfRows(inComponent: 0), 1)
            XCTAssertEqual($0[safe: 1]?.numberOfRows(inComponent: 0), 2)
            XCTAssertEqual($0[safe: 2]?.numberOfRows(inComponent: 0), 3)
            #endif
        }
    }
}
#endif
