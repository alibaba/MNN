#if !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class SliderTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformSlider = UISlider
    #elseif canImport(AppKit)
    typealias PlatformSlider = NSSlider
    #endif

    func testSlider() {
        XCTAssertViewIntrospection(of: PlatformSlider.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Slider(value: .constant(0.2), in: 0...1)
                    #if os(iOS)
                    .introspect(.slider, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.slider, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                Slider(value: .constant(0.5), in: 0...1)
                    #if os(iOS)
                    .introspect(.slider, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.slider, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                Slider(value: .constant(0.8), in: 0...1)
                    #if os(iOS)
                    .introspect(.slider, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.slider, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.value, 0.2)
            XCTAssertEqual($0[safe: 1]?.value, 0.5)
            XCTAssertEqual($0[safe: 2]?.value, 0.8)
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.floatValue, 0.2)
            XCTAssertEqual($0[safe: 1]?.floatValue, 0.5)
            XCTAssertEqual($0[safe: 2]?.floatValue, 0.8)
            #endif
        }
    }
}
#endif
