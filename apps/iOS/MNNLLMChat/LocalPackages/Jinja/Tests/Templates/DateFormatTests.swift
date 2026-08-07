//
//  DateFormatTests.swift
//  Jinja
//
//  Created by Sachin Desai on 5/13/25.
//

import XCTest
import OrderedCollections

@testable import Jinja

final class DateFormatTests: XCTestCase {

    // Fixed test date to ensure consistent results
    let testDate: Date = {
        var components = DateComponents()
        components.year = 2025
        components.month = 3
        components.day = 15
        components.hour = 14
        components.minute = 30
        components.second = 45
        components.nanosecond = 123456000

        let calendar = Calendar(identifier: .gregorian)
        return calendar.date(from: components)!
    }()

    // MARK: - Basic Format Tests

    func testBasicFormats() {
        // Test basic date components
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%B %d, %Y"), "March 15, 2025")
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%Y-%m-%d"), "2025-03-15")
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%H:%M:%S"), "14:30:45")
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%Y-%m-%d %H:%M:%S"), "2025-03-15 14:30:45")
    }

    // MARK: - Individual Format Specifier Tests

    func testWeekdayFormats() {
        // Test abbreviated weekday name
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%a"), "Sat")
        // Test full weekday name
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%A"), "Saturday")
        // Test weekday as a number (0-6, Sunday=0)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%w"), "6")
    }

    func testMonthFormats() {
        // Test month as zero-padded number
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%m"), "03")
        // Test month as non-padded number
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-m"), "3")
        // Test abbreviated month name
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%b"), "Mar")
        // Test full month name
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%B"), "March")
    }

    func testDayFormats() {
        // Test day of month as zero-padded number
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%d"), "15")
        // Test day of month as non-padded number
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-d"), "15")
        // Test day of year (001-366)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%j"), "074")
        // Test day of year without padding
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-j"), "74")
    }

    func testYearFormats() {
        // Test year with century
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%Y"), "2025")
        // Test year without century (00-99)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%y"), "25")
        // Test year without century and without padding
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-y"), "25")
    }

    func testTimeFormats() {
        // Test hour in 24-hour format (00-23)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%H"), "14")
        // Test hour without padding in 24-hour format
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-H"), "14")
        // Test hour in 12-hour format (01-12)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%I"), "02")
        // Test hour without padding in 12-hour format
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-I"), "2")
        // Test AM/PM
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%p"), "PM")
        // Test minute (00-59)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%M"), "30")
        // Test minute without padding
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-M"), "30")
        // Test second (00-59)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%S"), "45")
        // Test second without padding
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%-S"), "45")
        // Test microseconds
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%f"), "123456")
    }

    func testTimeZoneFormats() {
        // Note: These tests may need adjustment based on your test environment
        // Test timezone offset (e.g., +0000)
        // This is environment-dependent, so we'll just check format
        let tzOffsetResult = Environment.formatDate(testDate, withFormat: "%z")
        XCTAssertTrue(tzOffsetResult.hasPrefix("+") || tzOffsetResult.hasPrefix("-"))
        XCTAssertEqual(tzOffsetResult.count, 5)

        // Test timezone abbreviation (e.g., UTC, EST)
        // This is environment-dependent, so we'll just check it's not empty
        let tzAbbrResult = Environment.formatDate(testDate, withFormat: "%Z")
        XCTAssertFalse(tzAbbrResult.isEmpty)
    }

    func testWeekFormats() {
        // Test week number with Sunday as first day (00-53)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%U"), "11")
        // Test week number with Monday as first day (00-53)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%W"), "11")
    }

    func testCompleteFormats() {
        // Test locale's appropriate date and time representation
        let cFormatResult = Environment.formatDate(testDate, withFormat: "%c")
        XCTAssertFalse(cFormatResult.isEmpty)
        XCTAssertTrue(cFormatResult.contains("2025"))

        // Test locale's appropriate date representation
        let xFormatResult = Environment.formatDate(testDate, withFormat: "%x")
        XCTAssertFalse(xFormatResult.isEmpty)

        // Test locale's appropriate time representation
        let XFormatResult = Environment.formatDate(testDate, withFormat: "%X")
        XCTAssertFalse(XFormatResult.isEmpty)
    }

    // MARK: - Edge Cases

    func testEscapedPercent() {
        // Test escaped % character
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "100%%"), "100%")
    }

    func testUnknownFormatSpecifiers() {
        // Test unknown format specifiers (should pass through as-is)
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: "%Q"), "%Q")
    }

    func testEmptyFormat() {
        // Test empty format string
        XCTAssertEqual(Environment.formatDate(testDate, withFormat: ""), "")
    }

    func testComplexFormats() {
        // Test complex combinations of format specifiers
        XCTAssertEqual(
            Environment.formatDate(testDate, withFormat: "Date: %Y-%m-%d (%A) Time: %I:%M:%S %p"),
            "Date: 2025-03-15 (Saturday) Time: 02:30:45 PM"
        )
    }

    // MARK: - Special Cases

    func testSpecialDates() {
        // Test with January 1st (first day of year)
        var components = DateComponents()
        components.year = 2025
        components.month = 1
        components.day = 1
        components.hour = 0
        components.minute = 0
        components.second = 0

        let calendar = Calendar(identifier: .gregorian)
        let newYearsDay = calendar.date(from: components)!

        XCTAssertEqual(Environment.formatDate(newYearsDay, withFormat: "%Y-%m-%d"), "2025-01-01")
        XCTAssertEqual(Environment.formatDate(newYearsDay, withFormat: "%j"), "001")
        XCTAssertEqual(Environment.formatDate(newYearsDay, withFormat: "%A"), "Wednesday")
        XCTAssertEqual(Environment.formatDate(newYearsDay, withFormat: "%I:%M:%S %p"), "12:00:00 AM")

        // Test with December 31st (last day of year)
        components.month = 12
        components.day = 31
        components.hour = 23
        components.minute = 59
        components.second = 59

        let newYearsEve = calendar.date(from: components)!

        XCTAssertEqual(Environment.formatDate(newYearsEve, withFormat: "%Y-%m-%d"), "2025-12-31")
        XCTAssertEqual(Environment.formatDate(newYearsEve, withFormat: "%j"), "365")
        XCTAssertEqual(Environment.formatDate(newYearsEve, withFormat: "%A"), "Wednesday")
        XCTAssertEqual(Environment.formatDate(newYearsEve, withFormat: "%I:%M:%S %p"), "11:59:59 PM")
    }

    func testLeapYearDate() {
        // Test with February 29th on a leap year
        var components = DateComponents()
        components.year = 2024  // Leap year
        components.month = 2
        components.day = 29

        let calendar = Calendar(identifier: .gregorian)
        let leapYearDate = calendar.date(from: components)!

        XCTAssertEqual(Environment.formatDate(leapYearDate, withFormat: "%Y-%m-%d"), "2024-02-29")
        XCTAssertEqual(Environment.formatDate(leapYearDate, withFormat: "%j"), "060")  // Day of year
    }
}
