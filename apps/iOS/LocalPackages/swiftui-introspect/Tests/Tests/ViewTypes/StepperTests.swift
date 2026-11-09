#if !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class StepperTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformStepper = UIStepper
    #elseif canImport(AppKit)
    typealias PlatformStepper = NSStepper
    #endif

    func testStepper() {
        XCTAssertViewIntrospection(of: PlatformStepper.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                Stepper("", value: .constant(0), in: 0...10)
                    #if os(iOS)
                    .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                Stepper("", value: .constant(0), in: 0...10)
                    #if os(iOS)
                    .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                Stepper("", value: .constant(0), in: 0...10)
                    #if os(iOS)
                    .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            XCTAssert(Set($0.map(ObjectIdentifier.init)).count == 3)
        }
    }
}
#endif
