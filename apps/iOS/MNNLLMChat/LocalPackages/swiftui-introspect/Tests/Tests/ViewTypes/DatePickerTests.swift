#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class DatePickerTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformDatePicker = UIDatePicker
    #elseif canImport(AppKit)
    typealias PlatformDatePicker = NSDatePicker
    #endif

    func testDatePicker() {
        let date0 = Date(timeIntervalSince1970: 0)
        let date1 = Date(timeIntervalSince1970: 5)
        let date2 = Date(timeIntervalSince1970: 10)

        XCTAssertViewIntrospection(of: PlatformDatePicker.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                DatePicker("", selection: .constant(date0))
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.datePicker, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date1))
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.datePicker, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date2))
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.datePicker, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(UIKit)
            XCTAssertEqual($0[safe: 0]?.date, date0)
            XCTAssertEqual($0[safe: 1]?.date, date1)
            XCTAssertEqual($0[safe: 2]?.date, date2)
            #elseif canImport(AppKit)
            XCTAssertEqual($0[safe: 0]?.dateValue, date0)
            XCTAssertEqual($0[safe: 1]?.dateValue, date1)
            XCTAssertEqual($0[safe: 2]?.dateValue, date2)
            #endif
        }
    }
}
#endif
