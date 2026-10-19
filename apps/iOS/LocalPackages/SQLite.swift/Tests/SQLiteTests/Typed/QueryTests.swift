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

class QueryTests: XCTestCase {

    let users = Table("users")
    let id = Expression<Int64>("id")
    let email = Expression<String>("email")
    let age = Expression<Int?>("age")
    let admin = Expression<Bool>("admin")
    let optionalAdmin = Expression<Bool?>("admin")

    let posts = Table("posts")
    let userId = Expression<Int64>("user_id")
    let categoryId = Expression<Int64>("category_id")
    let published = Expression<Bool>("published")

    let categories = Table("categories")
    let tag = Expression<String>("tag")

    func test_select_withExpression_compilesSelectClause() {
        assertSQL("SELECT \"email\" FROM \"users\"", users.select(email))
    }

    func test_select_withStarExpression_compilesSelectClause() {
        assertSQL("SELECT * FROM \"users\"", users.select(*))
    }

    func test_select_withNamespacedStarExpression_compilesSelectClause() {
        assertSQL("SELECT \"users\".* FROM \"users\"", users.select(users[*]))
    }

    func test_select_withVariadicExpressions_compilesSelectClause() {
        assertSQL("SELECT \"email\", count(*) FROM \"users\"", users.select(email, count(*)))
    }

    func test_select_withExpressions_compilesSelectClause() {
        assertSQL("SELECT \"email\", count(*) FROM \"users\"", users.select([email, count(*)]))
    }

    func test_selectDistinct_withExpression_compilesSelectClause() {
        assertSQL("SELECT DISTINCT \"age\" FROM \"users\"", users.select(distinct: age))
    }

    func test_selectDistinct_withExpressions_compilesSelectClause() {
        assertSQL("SELECT DISTINCT \"age\", \"admin\" FROM \"users\"", users.select(distinct: [age, admin]))
    }

    func test_selectDistinct_withStar_compilesSelectClause() {
        assertSQL("SELECT DISTINCT * FROM \"users\"", users.select(distinct: *))
    }

    func test_union_compilesUnionClause() {
        assertSQL("SELECT * FROM \"users\" UNION SELECT * FROM \"posts\"", users.union(posts))
    }

    func test_union_compilesUnionAllClause() {
        assertSQL("SELECT * FROM \"users\" UNION ALL SELECT * FROM \"posts\"", users.union(all: true, posts))
    }

    func test_join_compilesJoinClause() {
        assertSQL(
            "SELECT * FROM \"users\" INNER JOIN \"posts\" ON (\"posts\".\"user_id\" = \"users\".\"id\")",
            users.join(posts, on: posts[userId] == users[id])
        )
    }

    func test_join_withExplicitType_compilesJoinClauseWithType() {
        assertSQL(
            "SELECT * FROM \"users\" LEFT OUTER JOIN \"posts\" ON (\"posts\".\"user_id\" = \"users\".\"id\")",
            users.join(.leftOuter, posts, on: posts[userId] == users[id])
        )

        assertSQL(
            "SELECT * FROM \"users\" CROSS JOIN \"posts\" ON (\"posts\".\"user_id\" = \"users\".\"id\")",
            users.join(.cross, posts, on: posts[userId] == users[id])
        )
    }

    func test_join_withTableCondition_compilesJoinClauseWithTableCondition() {
        assertSQL(
            "SELECT * FROM \"users\" INNER JOIN \"posts\" ON ((\"posts\".\"user_id\" = \"users\".\"id\") AND \"published\")",
            users.join(posts.filter(published), on: posts[userId] == users[id])
        )
    }

    func test_join_whenChained_compilesAggregateJoinClause() {
        assertSQL(
            "SELECT * FROM \"users\" " +
            "INNER JOIN \"posts\" ON (\"posts\".\"user_id\" = \"users\".\"id\") " +
            "INNER JOIN \"categories\" ON (\"categories\".\"id\" = \"posts\".\"category_id\")",
            users.join(posts, on: posts[userId] == users[id]).join(categories, on: categories[id] == posts[categoryId])
        )
    }

    func test_filter_compilesWhereClause() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 1)", users.filter(admin == true))
    }

    func test_filter_compilesWhereClause_false() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 0)", users.filter(admin == false))
    }

    func test_filter_compilesWhereClause_optional() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 1)", users.filter(optionalAdmin == true))
    }

    func test_filter_compilesWhereClause_optional_false() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 0)", users.filter(optionalAdmin == false))
    }

    func test_where_compilesWhereClause() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 1)", users.where(admin == true))
    }

    func test_where_compilesWhereClause_false() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 0)", users.where(admin == false))
    }

    func test_where_compilesWhereClause_optional() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 1)", users.where(optionalAdmin == true))
    }

    func test_where_compilesWhereClause_optional_false() {
        assertSQL("SELECT * FROM \"users\" WHERE (\"admin\" = 0)", users.where(optionalAdmin == false))
    }

    func test_filter_whenChained_compilesAggregateWhereClause() {
        assertSQL(
            "SELECT * FROM \"users\" WHERE ((\"age\" >= 35) AND \"admin\")",
            users.filter(age >= 35).filter(admin)
        )
    }

    func test_group_withSingleExpressionName_compilesGroupClause() {
        assertSQL("SELECT * FROM \"users\" GROUP BY \"age\"",
            users.group(age))
    }

    func test_group_withVariadicExpressionNames_compilesGroupClause() {
        assertSQL("SELECT * FROM \"users\" GROUP BY \"age\", \"admin\"", users.group(age, admin))
    }

    func test_group_withExpressionNameAndHavingBindings_compilesGroupClause() {
        assertSQL("SELECT * FROM \"users\" GROUP BY \"age\" HAVING \"admin\"", users.group(age, having: admin))
        assertSQL("SELECT * FROM \"users\" GROUP BY \"age\" HAVING (\"age\" >= 30)", users.group(age, having: age >= 30))
    }

    func test_group_withExpressionNamesAndHavingBindings_compilesGroupClause() {
        assertSQL(
            "SELECT * FROM \"users\" GROUP BY \"age\", \"admin\" HAVING \"admin\"",
            users.group([age, admin], having: admin)
        )
        assertSQL(
            "SELECT * FROM \"users\" GROUP BY \"age\", \"admin\" HAVING (\"age\" >= 30)",
            users.group([age, admin], having: age >= 30)
        )
    }

    func test_order_withSingleExpressionName_compilesOrderClause() {
        assertSQL("SELECT * FROM \"users\" ORDER BY \"age\"", users.order(age))
    }

    func test_order_withVariadicExpressionNames_compilesOrderClause() {
        assertSQL("SELECT * FROM \"users\" ORDER BY \"age\", \"email\"", users.order(age, email))
    }

    func test_order_withArrayExpressionNames_compilesOrderClause() {
        assertSQL("SELECT * FROM \"users\" ORDER BY \"age\", \"email\"", users.order([age, email]))
    }

    func test_order_withExpressionAndSortDirection_compilesOrderClause() {
//        AssertSQL("SELECT * FROM \"users\" ORDER BY \"age\" DESC, \"email\" ASC", users.order(age.desc, email.asc))
    }

    func test_order_whenChained_resetsOrderClause() {
        assertSQL("SELECT * FROM \"users\" ORDER BY \"age\"", users.order(email).order(age))
    }

    func test_reverse_withoutOrder_ordersByRowIdDescending() {
//        AssertSQL("SELECT * FROM \"users\" ORDER BY \"ROWID\" DESC", users.reverse())
    }

    func test_reverse_withOrder_reversesOrder() {
//        AssertSQL("SELECT * FROM \"users\" ORDER BY \"age\" DESC, \"email\" ASC", users.order(age, email.desc).reverse())
    }

    func test_limit_compilesLimitClause() {
        assertSQL("SELECT * FROM \"users\" LIMIT 5", users.limit(5))
    }

    func test_limit_withOffset_compilesOffsetClause() {
        assertSQL("SELECT * FROM \"users\" LIMIT 5 OFFSET 5", users.limit(5, offset: 5))
    }

    func test_limit_whenChained_overridesLimit() {
        let query = users.limit(5)

        assertSQL("SELECT * FROM \"users\" LIMIT 10", query.limit(10))
        assertSQL("SELECT * FROM \"users\"", query.limit(nil))
    }

    func test_limit_whenChained_withOffset_overridesOffset() {
        let query = users.limit(5, offset: 5)

        assertSQL("SELECT * FROM \"users\" LIMIT 10 OFFSET 20", query.limit(10, offset: 20))
        assertSQL("SELECT * FROM \"users\"", query.limit(nil))
    }

    func test_alias_aliasesTable() {
        let managerId = Expression<Int64>("manager_id")

        let managers = users.alias("managers")

        assertSQL(
            "SELECT * FROM \"users\" " +
            "INNER JOIN \"users\" AS \"managers\" ON (\"managers\".\"id\" = \"users\".\"manager_id\")",
            users.join(managers, on: managers[id] == users[managerId])
        )
    }

    func test_with_compilesWithClause() {
        let temp = Table("temp")

        assertSQL("WITH \"temp\" AS (SELECT * FROM \"users\") SELECT * FROM \"temp\"",
                  temp.with(temp, as: users))
    }

    func test_with_compilesWithRecursiveClause() {
        let temp = Table("temp")

        assertSQL("WITH RECURSIVE \"temp\" AS (SELECT * FROM \"users\") SELECT * FROM \"temp\"",
                  temp.with(temp, recursive: true, as: users))
    }

    func test_with_compilesWithMaterializedClause() {
        let temp = Table("temp")

        assertSQL("WITH \"temp\" AS MATERIALIZED (SELECT * FROM \"users\") SELECT * FROM \"temp\"",
                  temp.with(temp, hint: .materialized, as: users))
    }

    func test_with_compilesWithNotMaterializedClause() {
        let temp = Table("temp")

        assertSQL("WITH \"temp\" AS NOT MATERIALIZED (SELECT * FROM \"users\") SELECT * FROM \"temp\"",
                  temp.with(temp, hint: .notMaterialized, as: users))
    }

    func test_with_columns_compilesWithClause() {
        let temp = Table("temp")

        assertSQL("WITH \"temp\" (\"id\", \"email\") AS (SELECT * FROM \"users\") SELECT * FROM \"temp\"",
                  temp.with(temp, columns: [id, email], recursive: false, hint: nil, as: users))
    }

    func test_with_multiple_compilesWithClause() {
        let temp = Table("temp")
        let second = Table("second")
        let third = Table("third")

        let query = temp
            .with(temp, recursive: true, as: users)
            .with(second, recursive: true, as: posts)
            .with(third, hint: .materialized, as: categories)

        assertSQL(
            """
            WITH RECURSIVE \"temp\" AS (SELECT * FROM \"users\"),
             \"second\" AS (SELECT * FROM \"posts\"),
             \"third\" AS MATERIALIZED (SELECT * FROM \"categories\")
             SELECT * FROM \"temp\"
            """.replacingOccurrences(of: "\n", with: ""),
            query
        )
    }

    func test_insert_compilesInsertExpression() {
        assertSQL(
            "INSERT INTO \"users\" (\"email\", \"age\") VALUES ('alice@example.com', 30)",
            users.insert(email <- "alice@example.com", age <- 30)
        )
    }

    func test_insert_withOnConflict_compilesInsertOrOnConflictExpression() {
        assertSQL(
            "INSERT OR REPLACE INTO \"users\" (\"email\", \"age\") VALUES ('alice@example.com', 30)",
            users.insert(or: .replace, email <- "alice@example.com", age <- 30)
        )
    }

    func test_insert_compilesInsertExpressionWithDefaultValues() {
        assertSQL("INSERT INTO \"users\" DEFAULT VALUES", users.insert())
    }

    func test_insert_withQuery_compilesInsertExpressionWithSelectStatement() {
        let emails = Table("emails")

        assertSQL(
            "INSERT INTO \"emails\" SELECT \"email\" FROM \"users\" WHERE \"admin\"",
            emails.insert(users.select(email).filter(admin))
        )
    }

    func test_insert_many_compilesInsertManyExpression() {
        assertSQL(
            """
            INSERT INTO \"users\" (\"email\", \"age\") VALUES ('alice@example.com', 30), ('geoff@example.com', 32),
             ('alex@example.com', 83)
            """.replacingOccurrences(of: "\n", with: ""),
            users.insertMany([[email <- "alice@example.com", age <- 30],
                              [email <- "geoff@example.com", age <- 32], [email <- "alex@example.com", age <- 83]])
        )
    }
    func test_insert_many_compilesInsertManyNoneExpression() {
        assertSQL(
            "INSERT INTO \"users\" DEFAULT VALUES",
            users.insertMany([])
        )
    }

    func test_insert_many_withOnConflict_compilesInsertManyOrOnConflictExpression() {
        assertSQL(
            """
            INSERT OR REPLACE INTO \"users\" (\"email\", \"age\") VALUES ('alice@example.com', 30),
             ('geoff@example.com', 32), ('alex@example.com', 83)
            """.replacingOccurrences(of: "\n", with: ""),
            users.insertMany(or: .replace, [[email <- "alice@example.com", age <- 30],
                                            [email <- "geoff@example.com", age <- 32],
                                            [email <- "alex@example.com", age <- 83]])
        )
    }

    func test_insert_encodable() throws {
        let emails = Table("emails")
        let value = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let insert = try emails.insert(value)
        assertSQL(
            """
            INSERT INTO \"emails\" (\"int\", \"string\", \"bool\", \"float\", \"double\", \"date\", \"uuid\")
             VALUES (1, '2', 1, 3.0, 4.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F')
            """.replacingOccurrences(of: "\n", with: ""),
            insert
        )
    }

    #if !os(Linux) // depends on exact JSON serialization
    func test_insert_encodable_with_nested_encodable() throws {
        let emails = Table("emails")
        let value1 = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let value = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: "optional", sub: value1)
        let insert = try emails.insert(value)
        let encodedJSON = try JSONEncoder().encode(value1)
        let encodedJSONString = String(data: encodedJSON, encoding: .utf8)!

        let expectedSQL =
            """
            INSERT INTO \"emails\" (\"int\", \"string\", \"bool\", \"float\", \"double\", \"date\", \"uuid\", \"optional\",
             \"sub\") VALUES (1, '2', 1, 3.0, 4.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F',
             'optional', '\(encodedJSONString)')
            """.replacingOccurrences(of: "\n", with: "")

        // As JSON serialization gives a different result each time, we extract JSON and compare it by deserializing it
        // and keep comparing the query but with the json replaced by the `JSON` string
        let (expectedQuery, expectedJSON) = extractAndReplace(expectedSQL, regex: "\\{.*\\}", with: "JSON")
        let (actualQuery, actualJSON) = extractAndReplace(insert.asSQL(), regex: "\\{.*\\}", with: "JSON")
        XCTAssertEqual(expectedQuery, actualQuery)
        XCTAssertEqual(
            try JSONDecoder().decode(TestCodable.self, from: expectedJSON.data(using: .utf8)!),
            try JSONDecoder().decode(TestCodable.self, from: actualJSON.data(using: .utf8)!)
        )
    }
    #endif

    func test_insert_and_search_for_UUID() throws {
        struct Test: Codable {
            var uuid: UUID
            var string: String
        }
        let testUUID = UUID()
        let testValue = Test(uuid: testUUID, string: "value")
        let db = try Connection(.temporary)
        try db.run(table.create { t in
            t.column(uuid)
            t.column(string)
        }
        )

        let iQuery = try table.insert(testValue)
        try db.run(iQuery)

        let fQuery = table.filter(uuid == testUUID)
        if let result = try db.pluck(fQuery) {
            let testValueReturned = Test(uuid: result[uuid], string: result[string])
            XCTAssertEqual(testUUID, testValueReturned.uuid)
        } else {
            XCTFail("Search for uuid failed")
        }
    }

    func test_upsert_withOnConflict_compilesInsertOrOnConflictExpression() {
        assertSQL(
            """
            INSERT INTO \"users\" (\"email\", \"age\") VALUES ('alice@example.com', 30) ON CONFLICT (\"email\")
             DO UPDATE SET \"age\" = \"excluded\".\"age\"
            """.replacingOccurrences(of: "\n", with: ""),
            users.upsert(email <- "alice@example.com", age <- 30, onConflictOf: email)
        )
    }

    func test_upsert_encodable() throws {
        let emails = Table("emails")
        let string = Expression<String>("string")
        let value = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let insert = try emails.upsert(value, onConflictOf: string)
        assertSQL(
            """
            INSERT INTO \"emails\" (\"int\", \"string\", \"bool\", \"float\", \"double\", \"date\", \"uuid\")
             VALUES (1, '2', 1, 3.0, 4.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F') ON CONFLICT (\"string\")
             DO UPDATE SET \"int\" = \"excluded\".\"int\", \"bool\" = \"excluded\".\"bool\",
             \"float\" = \"excluded\".\"float\", \"double\" = \"excluded\".\"double\", \"date\" = \"excluded\".\"date\",
             \"uuid\" = \"excluded\".\"uuid\"
            """.replacingOccurrences(of: "\n", with: ""),
            insert
        )
    }

    func test_insert_many_encodables() throws {
        let emails = Table("emails")
        let value1 = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let value2 = TestCodable(int: 2, string: "3", bool: true, float: 3, double: 5,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: "optional", sub: nil)
        let value3 = TestCodable(int: 3, string: "4", bool: true, float: 3, double: 6,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let insert = try emails.insertMany([value1, value2, value3])
        assertSQL(
            """
            INSERT INTO \"emails\" (\"int\", \"string\", \"bool\", \"float\", \"double\", \"date\", \"uuid\", \"optional\", \"sub\")
             VALUES (1, '2', 1, 3.0, 4.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F', NULL, NULL),
             (2, '3', 1, 3.0, 5.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F', 'optional', NULL),
             (3, '4', 1, 3.0, 6.0, '1970-01-01T00:00:00.000', 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F', NULL, NULL)
            """.replacingOccurrences(of: "\n", with: ""),
            insert
        )
    }

    func test_update_compilesUpdateExpression() {
        assertSQL(
            "UPDATE \"users\" SET \"age\" = 30, \"admin\" = 1 WHERE (\"id\" = 1)",
            users.filter(id == 1).update(age <- 30, admin <- true)
        )
    }

    func test_update_compilesUpdateLimitOrderExpression() {
        assertSQL(
            "UPDATE \"users\" SET \"age\" = 30 ORDER BY \"id\" LIMIT 1",
            users.order(id).limit(1).update(age <- 30)
        )
    }

    func test_update_encodable() throws {
        let emails = Table("emails")
        let value = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let update = try emails.update(value)
        assertSQL(
            """
            UPDATE \"emails\" SET \"int\" = 1, \"string\" = '2', \"bool\" = 1, \"float\" = 3.0, \"double\" = 4.0,
             \"date\" = '1970-01-01T00:00:00.000', \"uuid\" = 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F'
            """.replacingOccurrences(of: "\n", with: ""),
            update
        )
    }

    func test_update_encodable_with_nested_encodable() throws {
        let emails = Table("emails")
        let value1 = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                 date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: nil)
        let value = TestCodable(int: 1, string: "2", bool: true, float: 3, double: 4,
                                date: Date(timeIntervalSince1970: 0), uuid: testUUIDValue, optional: nil, sub: value1)
        let update = try emails.update(value)

        // NOTE: As Linux JSON decoding doesn't order keys the same way, we need to check prefix, suffix,
        // and extract JSON to decode it and check the decoded object.

        let expectedPrefix =
            """
            UPDATE \"emails\" SET \"int\" = 1, \"string\" = '2', \"bool\" = 1, \"float\" = 3.0, \"double\" = 4.0,
             \"date\" = '1970-01-01T00:00:00.000', \"uuid\" = 'E621E1F8-C36C-495A-93FC-0C247A3E6E5F', \"sub\" = '
            """.replacingOccurrences(of: "\n", with: "")
        let expectedSuffix = "'"

        let sql = update.asSQL()
        XCTAssert(sql.hasPrefix(expectedPrefix))
        XCTAssert(sql.hasSuffix(expectedSuffix))

        let extractedJSON = String(sql[
            sql.index(sql.startIndex, offsetBy: expectedPrefix.count) ..<
            sql.index(sql.endIndex, offsetBy: -expectedSuffix.count)
        ])
        let decodedJSON = try JSONDecoder().decode(TestCodable.self, from: extractedJSON.data(using: .utf8)!)
        XCTAssertEqual(decodedJSON, value1)
    }

    func test_delete_compilesDeleteExpression() {
        assertSQL(
            "DELETE FROM \"users\" WHERE (\"id\" = 1)",
            users.filter(id == 1).delete()
        )
    }

    func test_delete_compilesDeleteLimitOrderExpression() {
        assertSQL(
            "DELETE FROM \"users\" ORDER BY \"id\" LIMIT 1",
            users.order(id).limit(1).delete()
        )
    }

    func test_delete_compilesExistsExpression() {
        assertSQL(
            "SELECT EXISTS (SELECT * FROM \"users\")",
            users.exists
        )
    }

    func test_count_returnsCountExpression() {
        assertSQL("SELECT count(*) FROM \"users\"", users.count)
    }

    func test_scalar_returnsScalarExpression() {
        assertSQL("SELECT \"int\" FROM \"table\"", table.select(int) as ScalarQuery<Int>)
        assertSQL("SELECT \"intOptional\" FROM \"table\"", table.select(intOptional) as ScalarQuery<Int?>)
        assertSQL("SELECT DISTINCT \"int\" FROM \"table\"", table.select(distinct: int) as ScalarQuery<Int>)
        assertSQL("SELECT DISTINCT \"intOptional\" FROM \"table\"", table.select(distinct: intOptional) as ScalarQuery<Int?>)
    }

    func test_subscript_withExpression_returnsNamespacedExpression() {
        let query = Table("query")

        assertSQL("\"query\".\"blob\"", query[data])
        assertSQL("\"query\".\"blobOptional\"", query[dataOptional])

        assertSQL("\"query\".\"bool\"", query[bool])
        assertSQL("\"query\".\"boolOptional\"", query[boolOptional])

        assertSQL("\"query\".\"date\"", query[date])
        assertSQL("\"query\".\"dateOptional\"", query[dateOptional])

        assertSQL("\"query\".\"double\"", query[double])
        assertSQL("\"query\".\"doubleOptional\"", query[doubleOptional])

        assertSQL("\"query\".\"int\"", query[int])
        assertSQL("\"query\".\"intOptional\"", query[intOptional])

        assertSQL("\"query\".\"int64\"", query[int64])
        assertSQL("\"query\".\"int64Optional\"", query[int64Optional])

        assertSQL("\"query\".\"string\"", query[string])
        assertSQL("\"query\".\"stringOptional\"", query[stringOptional])

        assertSQL("\"query\".*", query[*])
    }

    func test_tableNamespacedByDatabase() {
        let table = Table("table", database: "attached")

        assertSQL("SELECT * FROM \"attached\".\"table\"", table)
    }

}
