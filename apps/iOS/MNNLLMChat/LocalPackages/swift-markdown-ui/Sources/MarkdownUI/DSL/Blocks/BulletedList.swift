import Foundation

/// A Markdown bulleted list element.
///
/// You can create a bulleted list from a collection of elements.
///
/// ```swift
/// let headgear = ["Hats", "Caps", "Bonnets"]
///
/// var body: some View {
///   Markdown {
///     Paragraph {
///       "This is an incomplete list of headgear:"
///     }
///     BulletedList(of: headgear)
///   }
/// }
/// ```
///
/// ![](BulletedList)
///
/// To create a bulleted list from static items, you provide the items directly rather than the contents of a collection.
///
/// ```swift
/// Markdown {
///   Paragraph {
///     "This is an incomplete list of headgear:"
///   }
///   BulletedList {
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
///   Paragraph {
///     "List of unusual units of measurement:"
///   }
///   BulletedList {
///     ListItem {
///       "Length"
///       BulletedList {
///         "Hammer unit"
///         "Rack unit"
///         "Hand"
///         ListItem {
///           "Earth"
///           BulletedList {
///             "Radius"
///           }
///         }
///       }
///     }
///   }
/// }
/// ```
///
/// ![](NestedBulletedList)
public struct BulletedList: MarkdownContentProtocol {
  public var _markdownContent: MarkdownContent {
    .init(blocks: [.bulletedList(isTight: self.tight, items: self.items)])
  }

  private let tight: Bool
  private let items: [RawListItem]

  init(tight: Bool, items: [ListItem]) {
    // Force loose spacing if any of the items contains more than one paragraph
    let hasItemsWithMultipleParagraphs = items.contains { item in
      item.children.filter(\.isParagraph).count > 1
    }

    self.tight = hasItemsWithMultipleParagraphs ? false : tight
    self.items = items.map(\.children).map(RawListItem.init)
  }

  /// Creates a bulleted list with the specified items.
  /// - Parameters:
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - items: A list content builder that returns the items included in the list.
  public init(tight: Bool = true, @ListContentBuilder items: () -> [ListItem]) {
    self.init(tight: tight, items: items())
  }

  /// Creates a bulleted list from a sequence of elements.
  /// - Parameters:
  ///   - sequence: The sequence of elements.
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  ///   - items: A list content builder that returns the items for each element in the sequence.
  public init<S: Sequence>(
    of sequence: S,
    tight: Bool = true,
    @ListContentBuilder items: (S.Element) -> [ListItem]
  ) {
    self.init(tight: tight, items: sequence.flatMap { items($0) })
  }

  /// Creates a bulleted list from a sequence of strings.
  /// - Parameters:
  ///   - sequence: The sequence of strings.
  ///   - tight: A `Boolean` value that indicates if the list is tight or loose. This parameter is ignored if
  ///            any of the list items contain more than one paragraph.
  public init<S: Sequence>(of sequence: S, tight: Bool = true) where S.Element == String {
    self.init(tight: tight, items: sequence.map(ListItem.init(_:)))
  }
}
