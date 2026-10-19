import SwiftUI

extension View {
  func readWidth(column: Int) -> some View {
    self.background(
      GeometryReader { proxy in
        Color.clear.preference(key: ColumnWidthPreference.self, value: [column: proxy.size.width])
      }
    )
  }

  func onColumnWidthChange(perform action: @escaping ([Int: CGFloat]) -> Void) -> some View {
    self.onPreferenceChange(ColumnWidthPreference.self, perform: action)
  }
}

private struct ColumnWidthPreference: PreferenceKey {
  static let defaultValue: [Int: CGFloat] = [:]

  static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) {
    value.merge(nextValue(), uniquingKeysWith: max)
  }
}
