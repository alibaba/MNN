#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ButtonTests: XCTestCase {
    #if canImport(AppKit)
    typealias PlatformButton = NSButton
    #endif

    func testButton() {
        XCTAssertViewIntrospection(of: PlatformButton.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]
            let spy3 = spies[3]

            VStack {
                Button("Button 0", action: {})
                    .buttonStyle(.bordered)
                    #if os(macOS)
                    .introspect(.button, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                Button("Button 1", action: {})
                    .buttonStyle(.borderless)
                    #if os(macOS)
                    .introspect(.button, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                Button("Button 2", action: {})
                    .buttonStyle(.link)
                    #if os(macOS)
                    .introspect(.button, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif

                Button("Button 3", action: {})
                    #if os(macOS)
                    .introspect(.button, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy3)
                    #endif
            }
        } extraAssertions: {
            #if canImport(AppKit)
            XCTAssert(Set($0.map(ObjectIdentifier.init)).count == 4)
            #endif
        }
    }
}
#endif
