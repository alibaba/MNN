#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, macOS 11, *)
@MainActor
final class TextEditorTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformTextEditor = UITextView
    #elseif canImport(AppKit)
    typealias PlatformTextEditor = NSTextView
    #endif

    func testTextEditor() throws {
        guard #available(iOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformTextEditor.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TextEditor(text: .constant("Text Field 0"))
                    #if os(iOS) || os(visionOS)
                    .introspect(.textEditor, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.textEditor, on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                TextEditor(text: .constant("Text Field 1"))
                    #if os(iOS) || os(visionOS)
                    .introspect(.textEditor, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.textEditor, on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                TextEditor(text: .constant("Text Field 2"))
                    #if os(iOS) || os(visionOS)
                    .introspect(.textEditor, on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.textEditor, on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.text, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.text, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.text, "Text Field 2")
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.string, "Text Field 0")
            XCTAssertEqual($0[safe: 1]?.string, "Text Field 1")
            XCTAssertEqual($0[safe: 2]?.string, "Text Field 2")
            #endif
        }
    }
}
#endif
