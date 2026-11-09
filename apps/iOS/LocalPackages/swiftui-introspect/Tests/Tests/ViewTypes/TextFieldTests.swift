import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class TextFieldTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformTextField = UITextField
    #elseif canImport(AppKit)
    typealias PlatformTextField = NSTextField
    #endif

    func testTextField() {
        XCTAssertViewIntrospection(of: PlatformTextField.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TextField("", text: .constant("Text Field 0"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                TextField("", text: .constant("Text Field 1"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                TextField("", text: .constant("Text Field 2"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.text, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.text, "Text Field 2")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.stringValue, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.stringValue, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.stringValue, "Text Field 2")
            #endif
        }
    }

    func testTextFieldsEmbeddedInList() {
        XCTAssertViewIntrospection(of: PlatformTextField.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            List {
                TextField("", text: .constant("Text Field 0"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                TextField("", text: .constant("Text Field 1"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                TextField("", text: .constant("Text Field 2"))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.textField, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.text, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.text, "Text Field 2")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.stringValue, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.stringValue, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.stringValue, "Text Field 2")
            #endif
        }
    }
}
