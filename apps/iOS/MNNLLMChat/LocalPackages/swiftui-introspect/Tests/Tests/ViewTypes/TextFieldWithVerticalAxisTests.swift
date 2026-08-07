import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 16, tvOS 16, macOS 13, *)
@MainActor
final class TextFieldWithVerticalAxisTests: XCTestCase {
    #if canImport(UIKit) && (os(iOS) || os(visionOS))
    typealias PlatformTextField = UITextView
    #elseif canImport(UIKit) && os(tvOS)
    typealias PlatformTextField = UITextField
    #elseif canImport(AppKit)
    typealias PlatformTextField = NSTextField
    #endif

    func testTextFieldWithVerticalAxis() throws {
        guard #available(iOS 16, tvOS 16, macOS 13, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformTextField.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TextField("", text: .constant("Text Field 1"), axis: .vertical)
                    #if os(iOS) || os(visionOS)
                    .introspect(.textField(axis: .vertical), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(tvOS)
                    .introspect(.textField(axis: .vertical), on: .tvOS(.v16, .v17, .v18), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.textField(axis: .vertical), on: .macOS(.v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                TextField("", text: .constant("Text Field 2"), axis: .vertical)
                    #if os(iOS) || os(visionOS)
                    .introspect(.textField(axis: .vertical), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(tvOS)
                    .introspect(.textField(axis: .vertical), on: .tvOS(.v16, .v17, .v18), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.textField(axis: .vertical), on: .macOS(.v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                TextField("", text: .constant("Text Field 3"), axis: .vertical)
                    #if os(iOS) || os(visionOS)
                    .introspect(.textField(axis: .vertical), on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(tvOS)
                    .introspect(.textField(axis: .vertical), on: .tvOS(.v16, .v17, .v18), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.textField(axis: .vertical), on: .macOS(.v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Text Field 1")
            XCTAssertEqual($0[safe: 1]?.text, "Text Field 2")
            XCTAssertEqual($0[safe: 2]?.text, "Text Field 3")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.stringValue, "Text Field 1")
            XCTAssertEqual($0[safe: 1]?.stringValue, "Text Field 2")
            XCTAssertEqual($0[safe: 2]?.stringValue, "Text Field 3")
            #endif
        }
    }
}
