import SwiftUI

struct TaskListItemView: View {
  @Environment(\.theme.listItem) private var listItem
  @Environment(\.theme.taskListMarker) private var taskListMarker

  private let item: RawTaskListItem

  init(item: RawTaskListItem) {
    self.item = item
  }

  var body: some View {
    self.listItem.makeBody(
      configuration: .init(
        label: .init(self.label),
        content: .init(blocks: item.children)
      )
    )
  }

  private var label: some View {
    Label {
      BlockSequence(self.item.children)
    } icon: {
      self.taskListMarker.makeBody(configuration: .init(isCompleted: self.item.isCompleted))
        .textStyleFont()
    }
  }
}
