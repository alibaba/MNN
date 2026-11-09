#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ToggleWithCheckboxStyleTests: XCTestCase {
    #if canImport(AppKit) && !targetEnvironment(macCatalyst)
    typealias PlatformToggleWithCheckboxStyle = NSButton
    #endif

    func testToggleWithCheckboxStyle() throws {
        XCTAssertViewIntrospection(of: PlatformToggleWithCheckboxStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Toggle("", isOn: .constant(true))
                    .toggleStyle(.checkbox)
                    #if os(macOS)
                    .introspect(.toggle(style: .checkbox), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                Toggle("", isOn: .constant(false))
                    .toggleStyle(.checkbox)
                    #if os(macOS)
                    .introspect(.toggle(style: .checkbox), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                Toggle("", isOn: .constant(true))
                    .toggleStyle(.checkbox)
                    #if os(macOS)
                    .introspect(.toggle(style: .checkbox), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(AppKit) && !targetEnvironment(macCatalyst)
            XCTAssertEqual($0[safe: 0]?.state, .on)
            XCTAssertEqual($0[safe: 1]?.state, .off)
            XCTAssertEqual($0[safe: 2]?.state, .on)
            #endif
        }
    }
}
#endif
