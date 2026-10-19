import MarkdownUI
import XCTest

final class MarkdownContentTests: XCTestCase {
  func testEmpty() {
    // when
    let content = MarkdownContent("")

    // then
    XCTAssertEqual(MarkdownContent {}, content)
    XCTAssertEqual("", content.renderMarkdown())
  }

  func testBlockquote() {
    // given
    let markdown = """
      > Hello
      >\u{20}
      > > World
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Blockquote {
          "Hello"
          Blockquote {
            "World"
          }
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testList() {
    // given
    let markdown = """
      1.  one
      2.  two
            - nested 1
            - nested 2
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        NumberedList {
          "one"
          ListItem {
            "two"
            BulletedList {
              "nested 1"
              "nested 2"
            }
          }
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testLooseList() {
    // given
    let markdown = """
      9.  one

      10. two
      \u{20}\u{20}\u{20}\u{20}
            - nested 1
            - nested 2
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        NumberedList(tight: false, start: 9) {
          "one"
          ListItem {
            "two"
            BulletedList {
              "nested 1"
              "nested 2"
            }
          }
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testTaskList() {
    // given
    let markdown = """
      - [ ] one
      - [x] two
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        TaskList {
          "one"
          TaskListItem(isCompleted: true) {
            "two"
          }
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testCodeBlock() {
    // given
    let markdown = """
      ``` swift
      let a = 5
      let b = 42
      ```
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        CodeBlock(language: "swift") {
          """
          let a = 5
          let b = 42

          """
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testParagraph() {
    // given
    let markdown = "Hello world\\!"

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello world!"
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testHeading() {
    // given
    let markdown = """
      # Hello

      ## World
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Heading {
          "Hello"
        }
        Heading(.level2) {
          "World"
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testTable() throws {
    guard #available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *) else {
      throw XCTSkip("Required API is not available for this test")
    }

    // given
    let markdown = """
      |Default|Leading|Center|Trailing|
      | --- | :-- | :-: | --: |
      |git status|git status|git status|git status|
      |git diff|git diff|git diff|git diff|
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        TextTable {
          TextTableColumn<[String]>(title: "Default", value: \.[0])
          TextTableColumn(alignment: .leading, title: "Leading", value: \.[1])
          TextTableColumn(alignment: .center, title: "Center", value: \.[2])
          TextTableColumn(alignment: .trailing, title: "Trailing", value: \.[3])
        } rows: {
          TextTableRow(Array(repeating: "git status", count: 4))
          TextTableRow(Array(repeating: "git diff", count: 4))
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testThematicBreak() {
    // given
    let markdown = """
      Foo

      -----

      Bar
      """

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        "Foo"
        ThematicBreak()
        "Bar"
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testSoftBreak() {
    // given
    let markdown = "Hello\nWorld"

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello"
          SoftBreak()
          "World"
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testLineBreak() {
    // given
    let markdown = "Hello  \nWorld"

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello"
          LineBreak()
          "World"
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testCode() {
    // given
    let markdown = "Returns `nil`."

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Returns "
          Code("nil")
          "."
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testEmphasis() {
    // given
    let markdown = "Hello *world*."

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello "
          Emphasis("world")
          "."
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testStrong() {
    // given
    let markdown = "Hello **world**."

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello "
          Strong("world")
          "."
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testStrikethrough() {
    // given
    let markdown = "Hello ~~world~~."

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello "
          Strikethrough("world")
          "."
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testLink() {
    // given
    let markdown = "Hello [world](https://example.com)."

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          "Hello "
          InlineLink("world", destination: URL(string: "https://example.com")!)
          "."
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }

  func testImage() {
    // given
    let markdown = "![Puppy](https://picsum.photos/id/237/200/300)"

    // when
    let content = MarkdownContent(markdown)

    // then
    XCTAssertEqual(
      MarkdownContent {
        Paragraph {
          InlineImage("Puppy", source: URL(string: "https://picsum.photos/id/237/200/300")!)
        }
      },
      content
    )
    XCTAssertEqual(markdown, content.renderMarkdown())
  }
}
