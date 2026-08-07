import SwiftUI

@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
struct FlowLayout: Layout {
  let horizontalSpacing: CGFloat
  let verticalSpacing: CGFloat

  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
    let rows = self.computeLayout(for: proposal, subviews: subviews)
    return self.sizeThatFits(rows: rows)
  }

  func placeSubviews(
    in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void
  ) {
    let rows = self.computeLayout(for: proposal, subviews: subviews)
    var position = bounds.origin

    for row in rows {
      for item in row.items {
        // align to bottom
        let itemBounds = CGRect(origin: position, size: item.size)
          .offsetBy(dx: 0, dy: row.size.height - item.size.height)
        subviews[item.index].place(at: itemBounds.origin, proposal: .init(itemBounds.size))
        position.x += item.size.width + self.horizontalSpacing
      }

      position.x = bounds.origin.x
      position.y += row.size.height + self.verticalSpacing
    }
  }
}

@available(iOS 16.0, macOS 13.0, tvOS 16.0, watchOS 9.0, *)
extension FlowLayout {
  private struct Item {
    let index: Int
    let size: CGSize
  }

  private struct Row {
    var size: CGSize = .zero
    var items: [Item] = []
  }

  private func computeLayout(for proposal: ProposedViewSize, subviews: Subviews) -> [Row] {
    var rows: [Row] = []
    var currentRow = Row()

    for (index, view) in zip(subviews.indices, subviews) {
      // propose the remainder of the width for low prioriy views, otherwise the full width
      // this way we can use a spacer view for hard line breaks
      let proposedWidth =
        view.priority < 0 ? proposal.width.map { $0 - currentRow.size.width } : proposal.width
      let item = Item(
        index: index,
        size: view.sizeThatFits(.init(width: proposedWidth, height: nil))
      )

      if currentRow.size.width > 0,
        currentRow.size.width + item.size.width > (proposal.width ?? .infinity)
      {
        // Remove the spacing for the last item
        currentRow.size.width -= self.horizontalSpacing
        rows.append(currentRow)
        currentRow = Row()
      }

      currentRow.items.append(item)
      currentRow.size.width += item.size.width + self.horizontalSpacing
      currentRow.size.height = max(item.size.height, currentRow.size.height)
    }

    if !currentRow.items.isEmpty {
      // Remove the spacing for the last item
      currentRow.size.width -= self.horizontalSpacing
      rows.append(currentRow)
    }

    return rows
  }

  private func sizeThatFits(rows: [Row]) -> CGSize {
    zip(rows.indices, rows).reduce(CGSize.zero) { size, tuple in
      let (index, row) = tuple
      let spacing = index < rows.endIndex - 1 ? self.verticalSpacing : 0
      return CGSize(
        width: max(size.width, row.size.width),
        height: size.height + row.size.height + spacing
      )
    }
  }
}
