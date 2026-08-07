import Foundation

public enum QueryError: Error, CustomStringConvertible {
    case noSuchTable(name: String)
    case noSuchColumn(name: String, columns: [String])
    case ambiguousColumn(name: String, similar: [String])
    case unexpectedNullValue(name: String)

    public var description: String {
        switch self {
        case .noSuchTable(let name):
            return "No such table: \(name)"
        case .noSuchColumn(let name, let columns):
            return "No such column `\(name)` in columns \(columns)"
        case .ambiguousColumn(let name, let similar):
            return "Ambiguous column `\(name)` (please disambiguate: \(similar))"
        case .unexpectedNullValue(let name):
            return "Unexpected null value for column `\(name)`"
        }
    }
}
