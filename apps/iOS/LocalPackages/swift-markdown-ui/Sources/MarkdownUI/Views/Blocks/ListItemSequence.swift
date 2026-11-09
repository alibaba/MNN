import SwiftUI

struct ListItemSequence: View {
  private let items: [RawListItem]
  private let start: Int
  private let markerStyle: BlockStyle<ListMarkerConfiguration>
  private let markerWidth: CGFloat?

  init(
    items: [RawListItem],
    start: Int = 1,
    markerStyle: BlockStyle<ListMarkerConfiguration>,
    markerWidth: CGFloat? = nil
  ) {
    self.items = items
    self.start = start
    self.markerStyle = markerStyle
    self.markerWidth = markerWidth
  }

  var body: some View {
    BlockSequence(self.items) { index, item in
      ListItemView(
        item: item,
        number: self.start + index,
        markerStyle: self.markerStyle,
        markerWidth: self.markerWidth
      )
    }
    .labelStyle(.titleAndIcon)
  }
}
