import Foundation

/// A Markdown numbered list element.
///
/// You can create numbered lists from a collection of elements.
///
/// ```swift
/// let headgear = ["Hats", "Caps", "Bonnets"]
///
/// var body: some View {
///   Markdown {
///     Paragraph {
///       "This is an incomplete list of headgear:"
///     }
///     NumberedList(of: headgear)
///   }
/// }
/// ```
///
/// ![](NumberedList)
///
/// To create a numbered list from static items, you provide the items directly rather than a collection.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "This is an incomplete list of headgear:"
///   }
///   NumberedList {
///     "Hats"
///     "Caps"
///     "Bonnets"
///   }
/// }
/// ```
///
/// Use a ``ListItem`` element to nest a list inside a list item or have multi-paragraph list items.
///
/// ```swift
/// Markdown {
///   NumberedList {
///     ListItem {
///       "Item one"
///       "Additional paragraph"
///     }
///     ListItem {
///       "Item two"
///       BulletedList {
///         "Subitem one"
///         "Subitem two"
///       }
///     }
///   }
/// }
/// ```
///
/// ![](ListItem)
public struct NumberedList: MarkdownContentProtocol {
  public var _markdownContent: MarkdownContent {
    .init(blocks: [.numberedList(isTight: self.tight, start: self.start, items: self.items)])
  }

  private let tight: Bool
  private let start: Int
  private let items: [RawListItem]

  init(tight: Bool, start: Int, items: [ListItem]) {
    // Force loose spacing if any of the items contains more than one paragraph
    let hasItemsWithMultipleParagraphs = items.contains { item in
      item.children.filter(\.isParagraph).count > 1
    }
    self.tight = hasItemsWithMultipleParagraphs ? false : tight
    self.start = start
    self.items = items.map(\.children).map(RawListItem.init)
  }

  /// Creates a numbered list with the specified items.
  /// - Parameters:
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - start: The start number for this list.
  ///   - items: A list content builder that returns the items included in the list.
  public init(tight: Bool = true, start: Int = 1, @ListContentBuilder items: () -> [ListItem]) {
    self.init(tight: tight, start: start, items: items())
  }

  /// Creates a numbered list from a sequence of elements.
  /// - Parameters:
  ///   - sequence: The sequence of elements.
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - start: The start number for this list.
  ///   - items: A list content builder that returns the items for each element in the sequence.
  public init<S: Sequence>(
    of sequence: S,
    tight: Bool = true,
    start: Int = 1,
    @ListContentBuilder items: (S.Element) -> [ListItem]
  ) {
    self.init(tight: tight, start: start, items: sequence.flatMap { items($0) })
  }

  /// Creates a numbered list from a sequence of strings.
  /// - Parameters:
  ///   - sequence: The sequence of strings.
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - start: The start number for this list.
  public init<S: Sequence>(
    of sequence: S,
    tight: Bool = true,
    start: Int = 1
  ) where S.Element == String {
    self.init(tight: tight, start: start, items: sequence.map(ListItem.init(_:)))
  }
}
