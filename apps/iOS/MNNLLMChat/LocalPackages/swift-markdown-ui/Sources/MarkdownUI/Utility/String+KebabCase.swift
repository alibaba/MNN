import Foundation

extension String {
  func kebabCased() -> String {
    self.components(separatedBy: .alphanumerics.inverted)
      .map { $0.lowercased() }
      .joined(separator: "-")
  }
}
