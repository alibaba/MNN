#if !os(iOS) && !os(tvOS) && !os(visionOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class DatePickerWithStepperFieldStyleTests: XCTestCase {
    #if canImport(AppKit) && !targetEnvironment(macCatalyst)
    typealias PlatformDatePickerWithStepperFieldStyle = NSDatePicker
    #endif

    func testDatePickerWithStepperFieldStyle() {
        let date0 = Date(timeIntervalSince1970: 0)
        let date1 = Date(timeIntervalSince1970: 5)
        let date2 = Date(timeIntervalSince1970: 10)

        XCTAssertViewIntrospection(of: PlatformDatePickerWithStepperFieldStyle.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                DatePicker("", selection: .constant(date0))
                    .datePickerStyle(.stepperField)
                    #if os(macOS)
                    .introspect(.datePicker(style: .stepperField), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy0)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date1))
                    .datePickerStyle(.stepperField)
                    #if os(macOS)
                    .introspect(.datePicker(style: .stepperField), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy1)
                    #endif
                    .cornerRadius(8)

                DatePicker("", selection: .constant(date2))
                    .datePickerStyle(.stepperField)
                    #if os(macOS)
                    .introspect(.datePicker(style: .stepperField), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            #if canImport(AppKit) && !targetEnvironment(macCatalyst)
            XCTAssertEqual($0[safe: 0]?.dateValue, date0)
            XCTAssertEqual($0[safe: 1]?.dateValue, date1)
            XCTAssertEqual($0[safe: 2]?.dateValue, date2)
            #endif
        }
    }
}
#endif
