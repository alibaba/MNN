#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(macOS 12, *)
@MainActor
final class ToggleWithButtonStyleTests: XCTestCase {
    #if canImport(AppKit) && !targetEnvironment(macCatalyst)
    typealias PlatformToggleWithButtonStyle = NSButton
    #endif

    func testToggleWithButtonStyle() throws {
        guard #available(macOS 12, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformToggleWithButtonStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Toggle("", isOn: .constant(true))
                    .toggleStyle(.button)
                    #if os(macOS)
                    .introspect(.toggle(style: .button), on: .macOS(.v12, .v13, .v14), customize: spy0)
                    #endif

                Toggle("", isOn: .constant(false))
                    .toggleStyle(.button)
                    #if os(macOS)
                    .introspect(.toggle(style: .button), on: .macOS(.v12, .v13, .v14), customize: spy1)
                    #endif

                Toggle("", isOn: .constant(true))
                    .toggleStyle(.button)
                    #if os(macOS)
                    .introspect(.toggle(style: .button), on: .macOS(.v12, .v13, .v14), customize: spy2)
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
