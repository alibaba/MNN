#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, macOS 10.15.4, *)
@MainActor
final class DatePickerWithCompactStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformDatePickerWithCompactStyle = UIDatePicker
    #elseif canImport(AppKit)
    typealias PlatformDatePickerWithCompactStyle = NSDatePicker
    #endif

    func testDatePickerWithCompactStyle() throws {
        guard #available(iOS 14, macOS 10.15.4, *) else {
            throw XCTSkip()
        }

        let date0 = Date(timeIntervalSince1970: 0)
        let date1 = Date(timeIntervalSince1970: 5)
        let date2 = Date(timeIntervalSince1970: 10)

        XCTAssertViewIntrospection(of: PlatformDatePickerWithCompactStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                DatePicker("", selection: .constant(date0))
                    .datePickerStyle(.compact)
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker(style: .compact), on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.datePicker(style: .compact), on: .macOS(.v10_15_4, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date1))
                    .datePickerStyle(.compact)
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker(style: .compact), on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.datePicker(style: .compact), on: .macOS(.v10_15_4, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date2))
                    .datePickerStyle(.compact)
                    #if os(iOS) || os(visionOS)
                    .introspect(.datePicker(style: .compact), on: .iOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.datePicker(style: .compact), on: .macOS(.v10_15_4, .v11, .v12, .v13, .v14), customize: spy2)
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
