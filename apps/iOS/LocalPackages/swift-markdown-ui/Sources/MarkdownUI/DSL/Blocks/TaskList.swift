import Foundation

/// A Markdown task list element.
///
/// Task lists allow you to create lists of items with checkboxes.
///
/// You can create a task list from a collection of elements.
///
/// ```swift
/// struct MyTask {
///   var isCompleted: Bool = false
///   var title: String
/// }
///
/// let tasks = [
///   MyTask(isCompleted: true, title: "A finished task"),
///   MyTask(title: "An unfinished task"),
///   MyTask(title: "Another unfinished task"),
/// ]
///
/// var body: some View {
///   Markdown {
///     Paragraph {
///       "Things to do:"
///     }
///     TaskList(of: tasks) { task in
///       TaskListItem(isCompleted: task.isCompleted) {
///         Paragraph {
///           if task.isCompleted {
///             Strikethrough(task.title)
///           } else {
///             task.title
///           }
///         }
///       }
///     }
///   }
/// }
/// ```
///
/// ![](TaskList)
///
/// To create a task list from static items, you provide the items directly rather than a collection of data.
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
public struct TaskList: MarkdownContentProtocol {
  public var _markdownContent: MarkdownContent {
    .init(blocks: [.taskList(isTight: self.tight, items: self.items)])
  }

  private let tight: Bool
  private let items: [RawTaskListItem]

  init(tight: Bool, items: [TaskListItem]) {
    // Force loose spacing if any of the items contains more than one paragraph
    let hasItemsWithMultipleParagraphs = items.contains { item in
      item.children.filter(\.isParagraph).count > 1
    }

    self.tight = hasItemsWithMultipleParagraphs ? false : tight
    self.items = items.map {
      RawTaskListItem(isCompleted: $0.isCompleted, children: $0.children)
    }
  }

  /// Creates a task list with the given items.
  /// - Parameters:
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - items: A task list content builder that returns the items included in the list.
  public init(tight: Bool = true, @TaskListContentBuilder items: () -> [TaskListItem]) {
    self.init(tight: tight, items: items())
  }

  /// Creates a task list from a sequence of elements.
  /// - Parameters:
  ///   - sequence: The sequence of elements.
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - items: A task list content builder that returns the items for each element in the sequence.
  public init<S: Sequence>(
    of sequence: S,
    tight: Bool = true,
    @TaskListContentBuilder items: (S.Element) -> [TaskListItem]
  ) {
    self.init(tight: tight, items: sequence.flatMap { items($0) })
  }
}
