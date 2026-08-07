import Foundation

/// A Markdown task list item.
///
/// You can use task list items to compose task lists.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "Things to do:"
///   }
///   TaskList {
///     TaskListItem(isCompleted: true) {
///       Paragraph {
///         Strikethrough("A finished task")
///       }
///     }
///     TaskListItem {
///       "An unfinished task"
///     }
///     "Another unfinished task"
///   }
/// }
/// ```
public struct TaskListItem: Hashable {
  let isCompleted: Bool
  let children: [BlockNode]

  init(isCompleted: Bool, children: [BlockNode]) {
    self.isCompleted = isCompleted
    self.children = children
  }

  init(_ text: String) {
    self.init(isCompleted: false, children: [.paragraph(content: [.text(text)])])
  }

  public init(isCompleted: Bool = false, @MarkdownContentBuilder content: () -> MarkdownContent) {
    self.init(isCompleted: isCompleted, children: content().blocks)
  }
}
