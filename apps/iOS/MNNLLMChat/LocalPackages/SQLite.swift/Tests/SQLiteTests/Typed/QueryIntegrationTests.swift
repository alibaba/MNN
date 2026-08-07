import XCTest
#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif
@testable import SQLite

class QueryIntegrationTests: SQLiteTestCase {

    let id = Expression<Int64>("id")
    let email = Expression<String>("email")
    let age = Expression<Int>("age")

    override func setUpWithError() throws {
        try super.setUpWithError()
        try createUsersTable()
    }

    // MARK: -

    func test_select() throws {
        let managerId = Expression<Int64>("manager_id")
        let managers = users.alias("managers")

        let alice = try db.run(users.insert(email <- "alice@example.com"))
        _ = try db.run(users.insert(email <- "betsy@example.com", managerId <- alice))

        for user in try db.prepare(users.join(managers, on: managers[id] == users[managerId])) {
            _ = user[users[managerId]]
        }
    }

    func test_prepareRowIterator() throws {
        let names = ["a", "b", "c"]
        try insertUsers(names)

        let emailColumn = Expression<String>("email")
        let emails = try db.prepareRowIterator(users).map { $0[emailColumn] }

        XCTAssertEqual(names.map({ "\($0)@example.com" }), emails.sorted())
    }

    func test_ambiguousMap() throws {
        let names = ["a", "b", "c"]
        try insertUsers(names)

        let emails = try db.prepare("select email from users", []).map { $0[0] as! String  }

        XCTAssertEqual(names.map({ "\($0)@example.com" }), emails.sorted())
    }

    func test_select_optional() throws {
        let managerId = Expression<Int64?>("manager_id")
        let managers = users.alias("managers")

        let alice = try db.run(users.insert(email <- "alice@example.com"))
        _ = try db.run(users.insert(email <- "betsy@example.com", managerId <- alice))

        for user in try db.prepare(users.join(managers, on: managers[id] == users[managerId])) {
            _ = user[users[managerId]]
        }
    }

    func test_select_codable() throws {
        let table = Table("codable")
        try db.run(table.create { builder in
            builder.column(Expression<Int>("int"))
            builder.column(Expression<String>("string"))
            builder.column(Expression<Bool>("bool"))
            builder.column(Expression<Double>("float"))
            builder.column(Expression<Double>("double"))
            builder.column(Expression<Date>("date"))
            builder.column(Expression<UUID>("uuid"))
            builder.column(Expression<String?>("optional"))
            builder.column(Expression<Data>("sub"))
        })

        let value1 = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let value = TestCodable(int: 5, string: "6", bool: true, float: 7, double: 8,
                                date: Date(timeIntervalSince1970: 5000), uuid: testUUIDValue, optional: "optional", sub: value1)
        try db.run(table.insert(value))

        let rows = try db.prepare(table)
        let values: [TestCodable] = try rows.map({ try $0.decode() })
        XCTAssertEqual(values.count, 1)
        XCTAssertEqual(values[0].int, 5)
        XCTAssertEqual(values[0].string, "6")
        XCTAssertEqual(values[0].bool, true)
        XCTAssertEqual(values[0].float, 7)
        XCTAssertEqual(values[0].double, 8)
        XCTAssertEqual(values[0].date, Date(timeIntervalSince1970: 5000))
        XCTAssertEqual(values[0].uuid, testUUIDValue)
        XCTAssertEqual(values[0].optional, "optional")
        XCTAssertEqual(values[0].sub?.int, 1)
        XCTAssertEqual(values[0].sub?.string, "2")
        XCTAssertEqual(values[0].sub?.bool, true)
        XCTAssertEqual(values[0].sub?.float, 3)
        XCTAssertEqual(values[0].sub?.double, 4)
        XCTAssertEqual(values[0].sub?.date, Date(timeIntervalSince1970: 0))
        XCTAssertNil(values[0].sub?.optional)
        XCTAssertNil(values[0].sub?.sub)
    }

    func test_scalar() throws {
        XCTAssertEqual(0, try db.scalar(users.count))
        XCTAssertEqual(false, try db.scalar(users.exists))

        try insertUsers("alice")
        XCTAssertEqual(1, try db.scalar(users.select(id.average)))
    }

    func test_pluck() throws {
        let rowid = try db.run(users.insert(email <- "alice@example.com"))
        XCTAssertEqual(rowid, try db.pluck(users)![id])
    }

    func test_insert() throws {
        let id = try db.run(users.insert(email <- "alice@example.com"))
        XCTAssertEqual(1, id)
    }

    func test_insert_many() throws {
        let id = try db.run(users.insertMany([[email <- "alice@example.com"], [email <- "geoff@example.com"]]))
        XCTAssertEqual(2, id)
    }

    func test_insert_many_encodables() throws {
        let table = Table("codable")
        try db.run(table.create { builder in
            builder.column(Expression<Int?>("int"))
            builder.column(Expression<String?>("string"))
            builder.column(Expression<Bool?>("bool"))
            builder.column(Expression<Double?>("float"))
            builder.column(Expression<Double?>("double"))
            builder.column(Expression<Date?>("date"))
            builder.column(Expression<UUID?>("uuid"))
        })

        let value1 = TestOptionalCodable(int: 5, string: "6", bool: true, float: 7, double: 8,
                                         date: Date(timeIntervalSince1970: 5000), uuid: testUUIDValue)
        let valueWithNils = TestOptionalCodable(int: nil, string: nil, bool: nil, float: nil, double: nil, date: nil, uuid: nil)
        try db.run(table.insertMany([value1, valueWithNils]))

         let rows = try db.prepare(table)
         let values: [TestOptionalCodable] = try rows.map({ try $0.decode() })
         XCTAssertEqual(values.count, 2)
    }

    func test_insert_custom_encodable_type() throws {
        struct TestTypeWithOptionalArray: Codable {
            var myInt: Int
            var myString: String
            var myOptionalArray: [Int]?
        }

        let table = Table("custom_codable")
        try db.run(table.create { builder in
            builder.column(Expression<Int?>("myInt"))
            builder.column(Expression<String?>("myString"))
            builder.column(Expression<String?>("myOptionalArray"))
        })

        let customType = TestTypeWithOptionalArray(myInt: 13, myString: "foo", myOptionalArray: [1, 2, 3])
        try db.run(table.insert(customType))
        let rows = try db.prepare(table)
        let values: [TestTypeWithOptionalArray] = try rows.map({ try $0.decode() })
        XCTAssertEqual(values.count, 1, "return one optional custom type")

        let customTypeWithNil = TestTypeWithOptionalArray(myInt: 123, myString: "String", myOptionalArray: nil)
        try db.run(table.insert(customTypeWithNil))
        let rowsNil = try db.prepare(table)
        let valuesNil: [TestTypeWithOptionalArray] = try rowsNil.map({ try $0.decode() })
        XCTAssertEqual(valuesNil.count, 2, "return two custom objects, including one that contains a nil optional")
    }

    func test_upsert() throws {
        try XCTSkipUnless(db.satisfiesMinimumVersion(minor: 24))
        let fetchAge = { () throws -> Int? in
            try self.db.pluck(self.users.filter(self.email == "alice@example.com")).flatMap { $0[self.age] }
        }

        let id = try db.run(users.upsert(email <- "alice@example.com", age <- 30, onConflictOf: email))
        XCTAssertEqual(1, id)
        XCTAssertEqual(30, try fetchAge())

        let nextId = try db.run(users.upsert(email <- "alice@example.com", age <- 42, onConflictOf: email))
        XCTAssertEqual(1, nextId)
        XCTAssertEqual(42, try fetchAge())
    }

    func test_update() throws {
        let changes = try db.run(users.update(email <- "alice@example.com"))
        XCTAssertEqual(0, changes)
    }

    func test_delete() throws {
        let changes = try db.run(users.delete())
        XCTAssertEqual(0, changes)
    }

    func test_union() throws {
        let expectedIDs = [
            try db.run(users.insert(email <- "alice@example.com")),
            try db.run(users.insert(email <- "sally@example.com"))
        ]

        let query1 = users.filter(email == "alice@example.com")
        let query2 = users.filter(email == "sally@example.com")

        let actualIDs = try db.prepare(query1.union(query2)).map { $0[id] }
        XCTAssertEqual(expectedIDs, actualIDs)

        let query3 = users.select(users[*], Expression<Int>(literal: "1 AS weight")).filter(email == "sally@example.com")
        let query4 = users.select(users[*], Expression<Int>(literal: "2 AS weight")).filter(email == "alice@example.com")

        let sql = query3.union(query4).order(Expression<Int>(literal: "weight")).asSQL()
        XCTAssertEqual(sql,
        """
        SELECT "users".*, 1 AS weight FROM "users" WHERE ("email" = 'sally@example.com') UNION \
        SELECT "users".*, 2 AS weight FROM "users" WHERE ("email" = 'alice@example.com') ORDER BY weight
        """)

        let orderedIDs = try db.prepare(query3.union(query4).order(Expression<Int>(literal: "weight"), email)).map { $0[id] }
        XCTAssertEqual(Array(expectedIDs.reversed()), orderedIDs)
    }

    func test_no_such_column() throws {
        let doesNotExist = Expression<String>("doesNotExist")
        try insertUser("alice")
        let row = try db.pluck(users.filter(email == "alice@example.com"))!

        XCTAssertThrowsError(try row.get(doesNotExist)) { error in
            if case QueryError.noSuchColumn(let name, _) = error {
                XCTAssertEqual("\"doesNotExist\"", name)
            } else {
                XCTFail("unexpected error: \(error)")
            }
        }
    }

    func test_catchConstraintError() throws {
        try db.run(users.insert(email <- "alice@example.com"))
        do {
            try db.run(users.insert(email <- "alice@example.com"))
            XCTFail("expected error")
        } catch let Result.error(_, code, _) where code == SQLITE_CONSTRAINT {
            // expected
        } catch let error {
            XCTFail("unexpected error: \(error)")
        }
    }

    func test_extendedErrorCodes_catchConstraintError() throws {
        db.usesExtendedErrorCodes = true
        try db.run(users.insert(email <- "alice@example.com"))
        do {
            try db.run(users.insert(email <- "alice@example.com"))
            XCTFail("expected error")
        } catch let Result.extendedError(_, extendedCode, _) where extendedCode == 2_067 {
            // SQLITE_CONSTRAINT_UNIQUE expected
        } catch let error {
            XCTFail("unexpected error: \(error)")
        }
    }

    // https://github.com/stephencelis/SQLite.swift/issues/285
    func test_order_by_random() throws {
        try insertUsers(["a", "b", "c'"])
        let result = Array(try db.prepare(users.select(email).order(Expression<Int>.random()).limit(1)))
        XCTAssertEqual(1, result.count)
    }

    func test_with_recursive() throws {
        let nodes = Table("nodes")
        let id = Expression<Int64>("id")
        let parent = Expression<Int64?>("parent")
        let value = Expression<Int64>("value")

        try db.run(nodes.create { builder in
            builder.column(id)
            builder.column(parent)
            builder.column(value)
        })

        try db.run(nodes.insertMany([
            [id <- 0, parent <- nil, value <- 2],
            [id <- 1, parent <- 0, value <- 4],
            [id <- 2, parent <- 0, value <- 9],
            [id <- 3, parent <- 2, value <- 8],
            [id <- 4, parent <- 2, value <- 7],
            [id <- 5, parent <- 4, value <- 3]
        ]))

        // Compute the sum of the values of node 5 and its ancestors
        let ancestors = Table("ancestors")
        let sum = try db.scalar(
            ancestors
                .select(value.sum)
                .with(ancestors,
                      columns: [id, parent, value],
                      recursive: true,
                      as: nodes
                        .where(id == 5)
                        .union(all: true,
                               nodes.join(ancestors, on: nodes[id] == ancestors[parent])
                                    .select(nodes[id], nodes[parent], nodes[value])
                              )
                     )
        )

        XCTAssertEqual(21, sum)
    }

    /// Verify that `*` is properly expanded in a SELECT statement following a WITH clause.
    func test_with_glob_expansion() throws {
        let names = Table("names")
        let name = Expression<String>("name")
        try db.run(names.create { builder in
            builder.column(email)
            builder.column(name)
        })

        try db.run(users.insert(email <- "alice@example.com"))
        try db.run(names.insert(email <- "alice@example.com", name <- "Alice"))

        // WITH intermediate AS ( SELECT ... ) SELECT * FROM intermediate
        let intermediate = Table("intermediate")
        let rows = try db.prepare(
            intermediate
                .with(intermediate,
                      as: users
                        .select([id, users[email], name])
                        .join(names, on: names[email] == users[email])
                        .where(users[email] == "alice@example.com")
                     ))

        // There should be at least one row in the result.
        let row = try XCTUnwrap(rows.makeIterator().next())

        // Verify the column names
        XCTAssertEqual(row.columnNames.count, 3)
        XCTAssertNotNil(row[id])
        XCTAssertNotNil(row[name])
        XCTAssertNotNil(row[email])
    }

    func test_select_ntile_function() throws {
        let users = Table("users")

        try insertUser("Joey")
        try insertUser("Timmy")
        try insertUser("Jimmy")
        try insertUser("Billy")

        let bucket = ntile(1, id.asc)
        try db.prepare(users.select(id, bucket)).forEach {
            XCTAssertEqual($0[bucket], 1) // only 1 window
        }
    }

    func test_select_cume_dist_function() throws {
        let users = Table("users")

        try insertUser("Joey")
        try insertUser("Timmy")
        try insertUser("Jimmy")
        try insertUser("Billy")

        let cumeDist = cumeDist(email)
        let results = try db.prepare(users.select(id, cumeDist)).map {
            $0[cumeDist]
        }
        XCTAssertEqual([0.25, 0.5, 0.75, 1], results)
    }

    func test_select_window_row_number() throws {
        let users = Table("users")

        try insertUser("Billy")
        try insertUser("Jimmy")
        try insertUser("Joey")
        try insertUser("Timmy")

        let rowNumber = rowNumber(email.asc)
        var expectedRowNum = 1
        try db.prepare(users.select(id, rowNumber)).forEach {
            // should retrieve row numbers in order of INSERT above
            XCTAssertEqual($0[rowNumber], expectedRowNum)
            expectedRowNum += 1
        }
    }

    func test_select_window_ranking() throws {
        let users = Table("users")

        try insertUser("Billy")
        try insertUser("Jimmy")
        try insertUser("Joey")
        try insertUser("Timmy")

        let percentRank = percentRank(email)
        let actualPercentRank: [Int] = try db.prepare(users.select(id, percentRank)).map {
            Int($0[percentRank] * 100)
        }
        XCTAssertEqual([0, 33, 66, 100], actualPercentRank)

        let rank = rank(email)
        let actualRank: [Int] = try db.prepare(users.select(id, rank)).map {
            $0[rank]
        }
        XCTAssertEqual([1, 2, 3, 4], actualRank)

        let denseRank = denseRank(email)
        let actualDenseRank: [Int] = try db.prepare(users.select(id, denseRank)).map {
            $0[denseRank]
        }
        XCTAssertEqual([1, 2, 3, 4], actualDenseRank)
    }

    func test_select_window_values() throws {
        let users = Table("users")

        try insertUser("Billy")
        try insertUser("Jimmy")
        try insertUser("Joey")
        try insertUser("Timmy")

        let firstValue = email.firstValue(email.desc)
        try db.prepare(users.select(id, firstValue)).forEach {
            XCTAssertEqual($0[firstValue], "Timmy@example.com") // should grab last email alphabetically
        }

        let lastValue = email.lastValue(email.asc)
        var row = try db.pluck(users.select(id, lastValue))!
        XCTAssertEqual(row[lastValue], "Billy@example.com")

        let nthValue = email.value(1, email.asc)
        row = try db.pluck(users.select(id, nthValue))!
        XCTAssertEqual(row[nthValue], "Billy@example.com")
    }
}

extension Connection {
    func satisfiesMinimumVersion(minor: Int, patch: Int = 0) -> Bool {
        guard let version = try? scalar("SELECT sqlite_version()") as? String else { return false }
        let components = version.split(separator: ".", maxSplits: 3).compactMap { Int($0) }
        guard components.count == 3 else { return false }

        return components[1] >= minor && components[2] >= patch
    }
}
