import XCTest
import Foundation
import Dispatch
@testable import SQLite

#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

// https://github.com/stephencelis/SQLite.swift/issues/1071
#if !os(Linux)

class CustomAggregationTests: SQLiteTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
        try insertUser("Alice", age: 30, admin: true)
        try insertUser("Bob", age: 25, admin: true)
        try insertUser("Eve", age: 28, admin: false)
    }

    func testUnsafeCustomSum() throws {
        let step = { (bindings: [Binding?], state: UnsafeMutablePointer<Int64>) in
            if let v = bindings[0] as? Int64 {
                state.pointee += v
            }
        }

        let final = { (state: UnsafeMutablePointer<Int64>) -> Binding? in
            let v = state.pointee
            let p = UnsafeMutableBufferPointer(start: state, count: 1)
            p.deallocate()
            return v
        }
        db.createAggregation("mySUM1", step: step, final: final) {
            let v = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)
            v[0] = 0
            return v.baseAddress!
        }
        let result = try db.prepare("SELECT mySUM1(age) AS s FROM users")
        let i = result.columnNames.firstIndex(of: "s")!
        for row in result {
            let value = row[i] as? Int64
            XCTAssertEqual(83, value)
        }
    }

    func testUnsafeCustomSumGrouping() throws {
        let step = { (bindings: [Binding?], state: UnsafeMutablePointer<Int64>) in
            if let v = bindings[0] as? Int64 {
                state.pointee += v
            }
        }
        let final = { (state: UnsafeMutablePointer<Int64>) -> Binding? in
            let v = state.pointee
            let p = UnsafeMutableBufferPointer(start: state, count: 1)
            p.deallocate()
            return v
        }
        db.createAggregation("mySUM2", step: step, final: final) {
            let v = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)
            v[0] = 0
            return v.baseAddress!
        }
        let result = try db.prepare("SELECT mySUM2(age) AS s FROM users GROUP BY admin ORDER BY s")
        let i = result.columnNames.firstIndex(of: "s")!
        let values = result.compactMap { $0[i] as? Int64 }
        XCTAssertTrue(values.elementsEqual([28, 55]))
    }

    func testCustomSum() throws {
        let reduce: (Int64, [Binding?]) -> Int64 = { (last, bindings) in
            let v = (bindings[0] as? Int64) ?? 0
            return last + v
        }
        db.createAggregation("myReduceSUM1", initialValue: Int64(2000), reduce: reduce, result: { $0 })
        let result = try db.prepare("SELECT myReduceSUM1(age) AS s FROM users")
        let i = result.columnNames.firstIndex(of: "s")!
        for row in result {
            let value = row[i] as? Int64
            XCTAssertEqual(2083, value)
        }
    }

    func testCustomSumGrouping() throws {
        let reduce: (Int64, [Binding?]) -> Int64 = { (last, bindings) in
            let v = (bindings[0] as? Int64) ?? 0
            return last + v
        }
        db.createAggregation("myReduceSUM2", initialValue: Int64(3000), reduce: reduce, result: { $0 })
        let result = try db.prepare("SELECT myReduceSUM2(age) AS s FROM users GROUP BY admin ORDER BY s")
        let i = result.columnNames.firstIndex(of: "s")!
        let values = result.compactMap { $0[i] as? Int64 }
        XCTAssertTrue(values.elementsEqual([3028, 3055]))
    }

    func testCustomStringAgg() throws {
        let initial = String(repeating: " ", count: 64)
        let reduce: (String, [Binding?]) -> String = { (last, bindings) in
            let v = (bindings[0] as? String) ?? ""
            return last + v
        }
        db.createAggregation("myReduceSUM3", initialValue: initial, reduce: reduce, result: { $0 })
        let result = try db.prepare("SELECT myReduceSUM3(email) AS s FROM users")

        let i = result.columnNames.firstIndex(of: "s")!
        for row in result {
            let value = row[i] as? String
            XCTAssertEqual("\(initial)Alice@example.comBob@example.comEve@example.com", value)
        }
    }

    func testCustomObjectSum() throws {
        {
            let initial = TestObject(value: 1000)
            let reduce: (TestObject, [Binding?]) -> TestObject = { (last, bindings) in
                let v = (bindings[0] as? Int64) ?? 0
                return TestObject(value: last.value + v)
            }
            db.createAggregation("myReduceSUMX", initialValue: initial, reduce: reduce, result: { $0.value })
            // end this scope to ensure that the initial value is retained
            // by the createAggregation call.
            // swiftlint:disable:next trailing_semicolon
        }();

        {
            XCTAssertEqual(TestObject.inits, 1)
            let result = try! db.prepare("SELECT myReduceSUMX(age) AS s FROM users")
            let i = result.columnNames.firstIndex(of: "s")!
            for row in result {
                let value = row[i] as? Int64
                XCTAssertEqual(1083, value)
            }
        }()
        XCTAssertEqual(TestObject.inits, 4)
        XCTAssertEqual(TestObject.deinits, 3) // the initial value is still retained by the aggregate's state block, so deinits is one less than inits
    }
}
#endif

/// This class is used to test that aggregation state variables
/// can be reference types and are properly memory managed when
/// crossing the Swift<->C boundary multiple times.
class TestObject {
    static var inits = 0
    static var deinits = 0

    var value: Int64
    init(value: Int64) {
        self.value = value
        TestObject.inits += 1
    }
    deinit {
        TestObject.deinits += 1
    }
}
