import SwiftUI

/// A type that selects the visible borders on a Markdown table.
///
/// You use a table border selector to select the visible borders when creating a ``TableBorderStyle``.
public struct TableBorderSelector {
  var rectangles: (_ tableBounds: TableBounds, _ borderWidth: CGFloat) -> [CGRect]
}

extension TableBorderSelector {
  /// A table border selector that selects the outside borders of a table.
  public static var outsideBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      [tableBounds.bounds]
    }
  }

  /// A table border selector that selects the inside borders of a table.
  public static var insideBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      Self.insideHorizontalBorders.rectangles(tableBounds, borderWidth)
        + Self.insideVerticalBorders.rectangles(tableBounds, borderWidth)
    }
  }

  /// A table border selector that selects the inside horizontal borders of a table.
  public static var insideHorizontalBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      (0..<tableBounds.rowCount - 1)
        .map {
          tableBounds.bounds(forRow: $0)
            .insetBy(dx: -borderWidth, dy: -borderWidth)
        }
        .map {
          CGRect(
            origin: .init(x: $0.minX, y: $0.maxY - borderWidth),
            size: .init(width: $0.width, height: borderWidth)
          )
        }
    }
  }

  /// A table border selector that selects the inside vertical borders of a table.
  public static var insideVerticalBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      (0..<tableBounds.columnCount - 1)
        .map {
          tableBounds.bounds(forColumn: $0)
            .insetBy(dx: -borderWidth, dy: -borderWidth)
        }
        .map {
          CGRect(
            origin: .init(x: $0.maxX - borderWidth, y: $0.minY),
            size: .init(width: borderWidth, height: $0.height)
          )
        }
    }
  }

  /// A table border selector that selects the horizontal borders of a table.
  public static var horizontalBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      Self.outsideHorizontalBorders.rectangles(tableBounds, borderWidth)
        + Self.insideHorizontalBorders.rectangles(tableBounds, borderWidth)
    }
  }

  /// A table border selector that selects all the borders of a table.
  public static var allBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      Self.insideBorders.rectangles(tableBounds, borderWidth)
        + Self.outsideBorders.rectangles(tableBounds, borderWidth)
    }
  }
}

extension TableBorderSelector {
  fileprivate static var outsideHorizontalBorders: TableBorderSelector {
    TableBorderSelector { tableBounds, borderWidth in
      [
        CGRect(
          origin: .init(x: tableBounds.bounds.minX, y: tableBounds.bounds.minY),
          size: .init(width: tableBounds.bounds.width, height: borderWidth)
        ),
        CGRect(
          origin: .init(x: tableBounds.bounds.minX, y: tableBounds.bounds.maxY - borderWidth),
          size: .init(width: tableBounds.bounds.width, height: borderWidth)
        ),
      ]
    }
  }
}
