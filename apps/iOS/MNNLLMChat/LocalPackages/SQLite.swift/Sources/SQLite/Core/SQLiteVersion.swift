import Foundation

public struct SQLiteVersion: Comparable, CustomStringConvertible {
    public let major: Int
    public let minor: Int
    public var point: Int = 0

    public var description: String {
        "SQLite \(major).\(minor).\(point)"
    }

    public static func <(lhs: SQLiteVersion, rhs: SQLiteVersion) -> Bool {
        lhs.tuple < rhs.tuple
    }

    public static func ==(lhs: SQLiteVersion, rhs: SQLiteVersion) -> Bool {
        lhs.tuple == rhs.tuple
    }

    static var zero: SQLiteVersion = .init(major: 0, minor: 0)
    private var tuple: (Int, Int, Int) { (major, minor, point) }
}
