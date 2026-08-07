import MarkdownUI
import SwiftUI

struct ListsView: View {
  private let content = """
    You can make an unordered list by preceding one or more lines of text with `-`, `*`, or `+`.

    ```
    - George Washington
    - John Adams
    - Thomas Jefferson
    ```

    - George Washington
    - John Adams
    - Thomas Jefferson

    To order your list, precede each line with a number.

    ```
    1. James Madison
    2. James Monroe
    3. John Quincy Adams
    ```

    1. James Madison
    2. James Monroe
    3. John Quincy Adams

    ## Nested Lists

    You can create a nested list by indenting one or more list items below another item.

    ```
    1. First list item
       - First nested list item
         - Second nested list item
    ```

    1. First list item
       - First nested list item
         - Second nested list item

    ## Task lists

    To create a task list, preface list items with a hyphen and space followed by [ ].
    To mark a task as complete, use [x].

    ```
    - [x] Markdown rendering and styling
    - [ ] Documentation and sample code
    - [ ] Release MarkdownUI 2.0
    ```

    - [x] Markdown rendering and styling
    - [ ] Documentation and sample code
    - [ ] Release MarkdownUI 2.0

    Note that the `DocC` theme doesn't have a task list marker style and uses simple
    bullets.
    """

  private let customizedContent = """
    - George Washington
    - John Adams
    - Thomas Jefferson

    10. James Madison
    1. James Monroe
    1. John Quincy Adams

    - [x] Markdown rendering and styling
    - [ ] Documentation and sample code
    - [ ] Release MarkdownUI 2.0
    """

  var body: some View {
    DemoView {
      Markdown(self.content)

      Section("Customization Example") {
        Markdown(self.customizedContent)
      }
      .markdownBulletedListMarker(.dash)
      .markdownNumberedListMarker(.lowerRoman)
      .markdownBlockStyle(\.taskListMarker) { configuration in
        Image(systemName: configuration.isCompleted ? "checkmark.circle.fill" : "circle")
          .relativeFrame(minWidth: .em(1.5), alignment: .trailing)
      }
    }
  }
}

struct ListView_Previews: PreviewProvider {
  static var previews: some View {
    ListsView()
  }
}
