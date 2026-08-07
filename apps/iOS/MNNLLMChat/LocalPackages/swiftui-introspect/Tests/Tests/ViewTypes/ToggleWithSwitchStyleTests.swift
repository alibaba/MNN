#if !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ToggleWithSwitchStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformToggleWithSwitchStyle = UISwitch
    #elseif canImport(AppKit)
    typealias PlatformToggleWithSwitchStyle = NSSwitch
    #endif

    func testToggleWithSwitchStyle() {
        XCTAssertViewIntrospection(of: PlatformToggleWithSwitchStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Toggle("", isOn: .constant(true))
                    .toggleStyle(.switch)
                    #if os(iOS)
                    .introspect(.toggle(style: .switch), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.toggle(style: .switch), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                Toggle("", isOn: .constant(false))
                    .toggleStyle(.switch)
                    #if os(iOS)
                    .introspect(.toggle(style: .switch), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.toggle(style: .switch), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                Toggle("", isOn: .constant(true))
                    .toggleStyle(.switch)
                    #if os(iOS)
                    .introspect(.toggle(style: .switch), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.toggle(style: .switch), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.isOn, true)
            XCTAssertEqual($0[safe: 1]?.isOn, false)
            XCTAssertEqual($0[safe: 2]?.isOn, true)
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.state, .on)
            XCTAssertEqual($0[safe: 1]?.state, .off)
            XCTAssertEqual($0[safe: 2]?.state, .on)
            #endif
        }
    }
}
#endif
