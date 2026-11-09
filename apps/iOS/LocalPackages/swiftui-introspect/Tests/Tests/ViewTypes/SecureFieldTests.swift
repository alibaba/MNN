import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class SecureFieldTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformSecureField = UITextField
    #elseif canImport(AppKit)
    typealias PlatformSecureField = NSTextField
    #endif

    func testSecureField() {
        XCTAssertViewIntrospection(of: PlatformSecureField.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                SecureField("", text: .constant("Secure Field 0"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                SecureField("", text: .constant("Secure Field 1"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                SecureField("", text: .constant("Secure Field 2"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Secure Field 0")
            XCTAssertEqual($0[safe: 1]?.text, "Secure Field 1")
            XCTAssertEqual($0[safe: 2]?.text, "Secure Field 2")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.stringValue, "Secure Field 0")
            XCTAssertEqual($0[safe: 1]?.stringValue, "Secure Field 1")
            XCTAssertEqual($0[safe: 2]?.stringValue, "Secure Field 2")
            #endif
        }
    }

    func testSecureFieldsEmbeddedInList() {
        XCTAssertViewIntrospection(of: PlatformSecureField.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            List {
                SecureField("", text: .constant("Secure Field 0"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                SecureField("", text: .constant("Secure Field 1"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                SecureField("", text: .constant("Secure Field 2"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.secureField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.secureField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Secure Field 0")
            XCTAssertEqual($0[safe: 1]?.text, "Secure Field 1")
            XCTAssertEqual($0[safe: 2]?.text, "Secure Field 2")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.stringValue, "Secure Field 0")
            XCTAssertEqual($0[safe: 1]?.stringValue, "Secure Field 1")
            XCTAssertEqual($0[safe: 2]?.stringValue, "Secure Field 2")
            #endif
        }
    }
}
