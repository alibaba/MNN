import Foundation

struct TableDefinition: Equatable {
    let name: String
    let columns: [ColumnDefinition]
    let indexes: [IndexDefinition]

    var quotedColumnList: String {
        columns.map { $0.name.quote() }.joined(separator: ", ")
    }
}

// https://sqlite.org/schematab.html#interpretation_of_the_schema_table
public struct ObjectDefinition: Equatable {
    public enum ObjectType: String {
        case table, index, view, trigger
    }
    public let type: ObjectType

    // The name of the object
    public let name: String

    // The name of a table or view that the object is associated with.
    //  * For a table or view, a copy of the name column.
    //  * For an index, the name of the table that is indexed
    //  * For a trigger, the column stores the name of the table or view that causes the trigger to fire.
    public let tableName: String

    // The page number of the root b-tree page for tables and indexes, otherwise 0 or NULL
    public let rootpage: Int64

    // SQL text that describes the object (NULL for the internal indexes)
    public let sql: String?

    public var isInternal: Bool {
        name.starts(with: "sqlite_") || sql == nil
    }
}

// https://sqlite.org/syntax/column-def.html
// column-name -> type-name -> column-constraint*
public struct ColumnDefinition: Equatable {

    // The type affinity of a column is the recommended type for data stored in that column.
    // The important idea here is that the type is recommended, not required. Any column can still
    // store any type of data. It is just that some columns, given the choice, will prefer to use one
    // storage class over another. The preferred storage class for a column is called its "affinity".
    public enum Affinity: String, CustomStringConvertible, CaseIterable {
        case INTEGER
        case NUMERIC
        case REAL
        case TEXT
        case BLOB

        public var description: String {
            rawValue
        }

        init(_ string: String) {
            let test = string.uppercased()
            // https://sqlite.org/datatype3.html#determination_of_column_affinity
            if test.contains("INT") { // Rule 1
                self = .INTEGER
            } else if ["CHAR", "CLOB", "TEXT"].first(where: {test.contains($0)}) != nil { // Rule 2
                self = .TEXT
            } else if string.contains("BLOB") { // Rule 3
                self = .BLOB
            } else if ["REAL", "FLOA", "DOUB"].first(where: {test.contains($0)}) != nil { // Rule 4
                self = .REAL
            } else { // Rule 5
                self = .NUMERIC
            }
        }
    }

    public enum OnConflict: String, CaseIterable {
        case ROLLBACK
        case ABORT
        case FAIL
        case IGNORE
        case REPLACE

        init?(_ string: String) {
            guard let value = (OnConflict.allCases.first { $0.rawValue == string }) else { return nil }
            self = value
        }
    }

    public struct PrimaryKey: Equatable {
        let autoIncrement: Bool
        let onConflict: OnConflict?

        // swiftlint:disable:next force_try
        static let pattern = try! NSRegularExpression(pattern: "PRIMARY KEY\\s*(?:ASC|DESC)?\\s*(?:ON CONFLICT (\\w+)?)?\\s*(AUTOINCREMENT)?")

        init(autoIncrement: Bool = true, onConflict: OnConflict? = nil) {
            self.autoIncrement = autoIncrement
            self.onConflict = onConflict
        }

        init?(sql: String) {
            guard let match = PrimaryKey.pattern.firstMatch(
                in: sql,
                range: NSRange(location: 0, length: sql.count)) else {
                return nil
            }
            let conflict = match.range(at: 1)
            let onConflict: ColumnDefinition.OnConflict?
            if conflict.location != NSNotFound {
                onConflict = OnConflict((sql as NSString).substring(with: conflict))
            } else {
                onConflict = nil
            }
            let autoIncrement = match.range(at: 2).location != NSNotFound
            self.init(autoIncrement: autoIncrement, onConflict: onConflict)
        }
    }

    public struct ForeignKey: Equatable {
        let table: String
        let column: String
        let primaryKey: String?
        let onUpdate: String?
        let onDelete: String?
    }

    public let name: String
    public let primaryKey: PrimaryKey?
    public let type: Affinity
    public let nullable: Bool
    public let defaultValue: LiteralValue
    public let references: ForeignKey?

    public init(name: String,
                primaryKey: PrimaryKey? = nil,
                type: Affinity,
                nullable: Bool = true,
                defaultValue: LiteralValue = .NULL,
                references: ForeignKey? = nil) {
        self.name = name
        self.primaryKey = primaryKey
        self.type = type
        self.nullable = nullable
        self.defaultValue = defaultValue
        self.references = references
    }

    func rename(from: String, to: String) -> ColumnDefinition {
        guard from == name else { return self }
        return ColumnDefinition(name: to, primaryKey: primaryKey, type: type, nullable: nullable, defaultValue: defaultValue, references: references)
    }
}

public enum LiteralValue: Equatable, CustomStringConvertible {
    // swiftlint:disable force_try
    private static let singleQuote = try! NSRegularExpression(pattern: "^'(.*)'$")
    private static let doubleQuote = try! NSRegularExpression(pattern: "^\"(.*)\"$")
    private static let blob        = try! NSRegularExpression(pattern: "^[xX]\'(.*)\'$")
    // swiftlint:enable force_try

    case numericLiteral(String)
    case stringLiteral(String)
    // BLOB literals are string literals containing hexadecimal data and preceded by a single "x" or "X"
    // character. Example: X'53514C697465'
    case blobLiteral(String)

    // If there is no explicit DEFAULT clause attached to a column definition, then the default value of the
    // column is NULL
    case NULL

    // Beginning with SQLite 3.23.0 (2018-04-02), SQLite recognizes the identifiers "TRUE" and
    // "FALSE" as boolean literals, if and only if those identifiers are not already used for some other
    // meaning.
    //
    // The boolean identifiers TRUE and FALSE are usually just aliases for the integer values 1 and 0, respectively.
    case TRUE
    case FALSE
    // swiftlint:disable identifier_name
    case CURRENT_TIME
    case CURRENT_DATE
    case CURRENT_TIMESTAMP
    // swiftlint:enable identifier_name

    init(_ string: String?) {
        guard let string else {
            self = .NULL
            return
        }
        switch string {
        case "NULL": self = .NULL
        case "TRUE": self = .TRUE
        case "FALSE": self = .FALSE
        case "CURRENT_TIME": self = .CURRENT_TIME
        case "CURRENT_TIMESTAMP": self = .CURRENT_TIMESTAMP
        case "CURRENT_DATE": self = .CURRENT_DATE
        default: self = LiteralValue.parse(string)
        }
    }

    public var description: String {
        switch self {
        case .NULL: return "NULL"
        case .TRUE: return "TRUE"
        case .FALSE: return "FALSE"
        case .CURRENT_TIMESTAMP: return "CURRENT_TIMESTAMP"
        case .CURRENT_TIME: return "CURRENT_TIME"
        case .CURRENT_DATE: return "CURRENT_DATE"
        case .stringLiteral(let value): return value.quote("'")
        case .blobLiteral(let value): return "X\(value.quote("'"))"
        case .numericLiteral(let value): return value
        }
    }

    func map<U>(block: (LiteralValue) -> U) -> U? {
        if self == .NULL {
            return nil
        } else {
            return block(self)
        }
    }
    private static func parse(_ string: String) -> LiteralValue {
        if let match = LiteralValue.singleQuote.firstMatch(in: string, range: NSRange(location: 0, length: string.count)) {
            return .stringLiteral((string as NSString).substring(with: match.range(at: 1)).replacingOccurrences(of: "''", with: "'"))
        } else if let match = LiteralValue.doubleQuote.firstMatch(in: string, range: NSRange(location: 0, length: string.count)) {
            return .stringLiteral((string as NSString).substring(with: match.range(at: 1)).replacingOccurrences(of: "\"\"", with: "\""))
        } else if let match = LiteralValue.blob.firstMatch(in: string, range: NSRange(location: 0, length: string.count)) {
            return .blobLiteral((string as NSString).substring(with: match.range(at: 1)))
        } else {
            return .numericLiteral(string)
        }
    }
}

// https://sqlite.org/lang_createindex.html
// schema-name.index-name ON table-name ( indexed-column+ ) WHERE expr
public struct IndexDefinition: Equatable {
    // SQLite supports index names up to 64 characters.
    static let maxIndexLength = 64

    // swiftlint:disable force_try
    static let whereRe = try! NSRegularExpression(pattern: "\\sWHERE\\s+(.+)$")
    static let orderRe = try! NSRegularExpression(pattern: "\"?(\\w+)\"? DESC")
    // swiftlint:enable force_try

    public enum Order: String { case ASC, DESC }

    public init(table: String, name: String, unique: Bool = false, columns: [String], `where`: String? = nil, orders: [String: Order]? = nil) {
        self.table = table
        self.name = name
        self.unique = unique
        self.columns = columns
        self.where = `where`
        self.orders = orders
    }

    init (table: String, name: String, unique: Bool, columns: [String], indexSQL: String?) {
        func wherePart(sql: String) -> String? {
            IndexDefinition.whereRe.firstMatch(in: sql, options: [], range: NSRange(location: 0, length: sql.count)).map {
                (sql as NSString).substring(with: $0.range(at: 1))
            }
        }

        func orders(sql: String) -> [String: IndexDefinition.Order] {
            IndexDefinition.orderRe
                .matches(in: sql, range: NSRange(location: 0, length: sql.count))
                .reduce([String: IndexDefinition.Order]()) { (memo, result) in
                        var memo2 = memo
                        let column = (sql as NSString).substring(with: result.range(at: 1))
                        memo2[column] = .DESC
                        return memo2
            }
        }
        self.init(table: table,
                  name: name,
                  unique: unique,
                  columns: columns,
                  where: indexSQL.flatMap(wherePart),
                  orders: indexSQL.flatMap(orders))
    }

    public let table: String
    public let name: String
    public let unique: Bool
    public let columns: [String]
    public let `where`: String?
    public let orders: [String: Order]?

    enum IndexError: LocalizedError {
        case tooLong(String, String)

        var errorDescription: String? {
            switch self {
            case .tooLong(let name, let table):
                return "Index name '\(name)' on table '\(table)' is too long; the limit is " +
                     "\(IndexDefinition.maxIndexLength) characters"
            }
        }
    }

    func validate() throws {
        if name.count > IndexDefinition.maxIndexLength {
            throw IndexError.tooLong(name, table)
        }
    }
}

public struct ForeignKeyError: CustomStringConvertible {
    public let from: String
    public let rowId: Int64
    public let to: String

    public var description: String {
        "\(from) [\(rowId)] => \(to)"
    }
}

extension TableDefinition {
    func toSQL(temporary: Bool = false) -> String {
        precondition(columns.count > 0, "no columns to create")

        return ([
            "CREATE",
            temporary ? "TEMPORARY" : nil,
            "TABLE",
            name,
            "(",
            columns.map { $0.toSQL() }.joined(separator: ",\n"),
            ")"
        ] as [String?]).compactMap { $0 }
         .joined(separator: " ")
    }

    func copySQL(to: TableDefinition) -> String {
        precondition(columns.count > 0)
        precondition(columns.count == to.columns.count, "column counts don't match")
        return "INSERT INTO \(to.name.quote()) (\(to.quotedColumnList)) SELECT \(quotedColumnList) FROM \(name.quote())"
    }
}

extension ColumnDefinition {
    func toSQL() -> String {
        [
            name.quote(),
            type.rawValue,
            defaultValue.map { "DEFAULT \($0)" },
            primaryKey.map { $0.toSQL() },
            nullable ? nil : "NOT NULL",
            references.map { $0.toSQL() }
        ].compactMap { $0 }
         .joined(separator: " ")
    }
}

extension IndexDefinition {
    public func toSQL(ifNotExists: Bool = false) -> String {
        let commaSeparatedColumns = columns.map { (column: String) -> String in
            column.quote() + (orders?[column].map { " \($0.rawValue)" } ?? "")
        }.joined(separator: ", ")

        return ([
            "CREATE",
            unique ? "UNIQUE" : nil,
            "INDEX",
            ifNotExists ? "IF NOT EXISTS" : nil,
            name.quote(),
            "ON",
            table.quote(),
            "(\(commaSeparatedColumns))",
            `where`.map { "WHERE \($0)" }
        ] as [String?]).compactMap { $0 }
         .joined(separator: " ")
    }
}

extension ColumnDefinition.ForeignKey {
    func toSQL() -> String {
        ([
            "REFERENCES",
            table.quote(),
            primaryKey.map { "(\($0.quote()))" },
            onUpdate.map { "ON UPDATE \($0)" },
            onDelete.map { "ON DELETE \($0)" }
        ] as [String?]).compactMap { $0 }
         .joined(separator: " ")
    }
}

extension ColumnDefinition.PrimaryKey {
    func toSQL() -> String {
        [
            "PRIMARY KEY",
            autoIncrement ? "AUTOINCREMENT" : nil,
            onConflict.map { "ON CONFLICT \($0.rawValue)" }
        ].compactMap { $0 }.joined(separator: " ")
    }
}
