#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class PickerWithMenuStyleTests: XCTestCase {
    #if canImport(AppKit) && !targetEnvironment(macCatalyst)
    typealias PlatformPickerWithMenuStyle = NSPopUpButton
    #endif

    func testPickerWithMenuStyle() {
        XCTAssertViewIntrospection(of: PlatformPickerWithMenuStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                }
                .pickerStyle(.menu)
                #if os(macOS)
                .introspect(.picker(style: .menu), on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                }
                .pickerStyle(.menu)
                #if os(macOS)
                .introspect(.picker(style: .menu), on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                #endif
                .cornerRadius(8)

                Picker("Pick", selection: .constant("1")) {
                    Text("1").tag("1")
                    Text("2").tag("2")
                    Text("3").tag("3")
                }
                .pickerStyle(.menu)
                #if os(macOS)
                .introspect(.picker(style: .menu), on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                #endif
            }
        } extraAssertions: {
            #if canImport(AppKit) && !targetEnvironment(macCatalyst)
            XCTAssertEqual($0[safe: 0]?.numberOfItems, 1)
            XCTAssertEqual($0[safe: 1]?.numberOfItems, 2)
            XCTAssertEqual($0[safe: 2]?.numberOfItems, 3)
            #endif
        }
    }
}
#endif
