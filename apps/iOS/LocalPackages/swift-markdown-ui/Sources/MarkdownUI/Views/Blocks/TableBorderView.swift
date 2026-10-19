import SwiftUI

struct TableBorderView: View {
  @Environment(\.tableBorderStyle) private var tableBorderStyle

  private let tableBounds: TableBounds

  init(tableBounds: TableBounds) {
    self.tableBounds = tableBounds
  }

  var body: some View {
    ZStack(alignment: .topLeading) {
      let rectangles = self.tableBorderStyle.visibleBorders.rectangles(
        self.tableBounds, self.borderWidth
      )
      ForEach(0..<rectangles.count, id: \.self) {
        let rectangle = rectangles[$0]
        Rectangle()
          .strokeBorder(self.tableBorderStyle.color, style: self.tableBorderStyle.strokeStyle)
          .offset(x: rectangle.minX, y: rectangle.minY)
          .frame(width: rectangle.width, height: rectangle.height)
      }
    }
  }

  private var borderWidth: CGFloat {
    self.tableBorderStyle.strokeStyle.lineWidth
  }
}
