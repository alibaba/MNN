import SwiftUI

struct TaskListView: View {
  @Environment(\.theme.list) private var list
  @Environment(\.listLevel) private var listLevel

  private let isTight: Bool
  private let items: [RawTaskListItem]

  init(isTight: Bool, items: [RawTaskListItem]) {
    self.isTight = isTight
    self.items = items
  }

  var body: some View {
    self.list.makeBody(
      configuration: .init(
        label: .init(self.label),
        content: .init(block: .taskList(isTight: self.isTight, items: self.items))
      )
    )
  }

  private var label: some View {
    BlockSequence(self.items) { _, item in
      TaskListItemView(item: item)
    }
    .labelStyle(.titleAndIcon)
    .environment(\.listLevel, self.listLevel + 1)
    .environment(\.tightSpacingEnabled, self.isTight)
  }
}
