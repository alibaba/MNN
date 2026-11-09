#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, macOS 11, *)
@MainActor
final class ColorPickerTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformColor = UIColor
    typealias PlatformColorPicker = UIColorWell
    #elseif canImport(AppKit)
    typealias PlatformColor = NSColor
    typealias PlatformColorPicker = NSColorWell
    #endif

    func testColorPicker() throws {
        guard #available(iOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformColorPicker.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                ColorPicker("", selection: .constant(PlatformColor.red.cgColor))
                    #if os(iOS) || os(visionOS)
                    .introspect(.colorPicker, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.colorPicker, on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                ColorPicker("", selection: .constant(PlatformColor.green.cgColor))
                    #if os(iOS) || os(visionOS)
                    .introspect(.colorPicker, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.colorPicker, on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                ColorPicker("", selection: .constant(PlatformColor.blue.cgColor))
                    #if os(iOS) || os(visionOS)
                    .introspect(.colorPicker, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.colorPicker, on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.selectedColor, .red)
            XCTAssertEqual($0[safe: 1]?.selectedColor, .green)
            XCTAssertEqual($0[safe: 2]?.selectedColor, .blue)
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.color, .red)
            XCTAssertEqual($0[safe: 1]?.color, .green)
            XCTAssertEqual($0[safe: 2]?.color, .blue)
            #endif
        }
    }
}
#endif
