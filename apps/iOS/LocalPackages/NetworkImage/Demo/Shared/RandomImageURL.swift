import Foundation

extension URL {
  private static let identifiers = [
    "0", "1", "10", "11", "23", "26", "31",
    "34", "58", "63", "91", "100", "103",
    "119", "1000", "1001", "1002", "1003",
    "1004", "1005", "1006", "1008", "1009",
    "101", "1010", "1011", "1012", "1013",
    "1014", "1015", "1016", "1018", "1019",
    "102", "1020", "1021", "1022", "1023",
    "1024", "1025",
  ]

  static func randomImageURL(size: CGSize) -> URL {
    URL(
      string:
        "https://picsum.photos/id/\(self.identifiers.randomElement() ?? "0")/\(Int(size.width))/\(Int(size.height))"
    )!
  }
}
