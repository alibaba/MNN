//
//  TableViewCellManager.swift
//  MarkdownView
//
//  Created by ktiays on 2025/1/27.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import Litext
import UIKit

final class TableViewCellManager {
    // MARK: - Properties

    private(set) var cells: [LTXLabel] = []
    private(set) var cellSizes: [CGSize] = []
    private(set) var widths: [CGFloat] = []
    private(set) var heights: [CGFloat] = []
    private var theme: MarkdownTheme = .default

    // MARK: - Cell Configuration

    func configureCells(
        for contents: [[NSAttributedString]],
        in containerView: UIView,
        cellPadding: CGFloat,
        maximumCellWidth: CGFloat
    ) {
        let numberOfRows = contents.count
        let numberOfColumns = contents.first?.count ?? 0

        // Reset arrays
        cellSizes = Array(repeating: .zero, count: numberOfRows * numberOfColumns)
        cells.forEach { $0.removeFromSuperview() }
        cells.removeAll()
        widths = Array(repeating: 0, count: numberOfColumns)
        heights = Array(repeating: 0, count: numberOfRows)

        // Configure cells for each row and column
        for (row, rowContent) in contents.enumerated() {
            var rowHeight: CGFloat = 0

            for (column, cellString) in rowContent.enumerated() {
                let index = row * rowContent.count + column
                let isHeaderCell = row == 0
                let cell = createOrUpdateCell(
                    at: index,
                    with: cellString,
                    maximumWidth: maximumCellWidth,
                    isHeader: isHeaderCell,
                    in: containerView
                )

                let cellSize = calculateCellSize(for: cell, cellPadding: cellPadding)
                cellSizes[index] = cellSize

                // Update row and column dimensions
                rowHeight = max(rowHeight, cellSize.height)
                widths[column] = max(widths[column], cellSize.width)
            }

            heights[row] = rowHeight
        }
    }

    // MARK: - Public Methods

    func setTheme(_ theme: MarkdownTheme) {
        self.theme = theme
        updateCellsAppearance()
    }

    // MARK: - Private Methods

    private func createOrUpdateCell(
        at index: Int,
        with attributedText: NSAttributedString,
        maximumWidth: CGFloat,
        isHeader: Bool,
        in containerView: UIView
    ) -> LTXLabel {
        let cell: LTXLabel

        if index >= cells.count {
            cell = LTXLabel()
            cell.isSelectable = true
            cell.backgroundColor = .clear
            cell.preferredMaxLayoutWidth = maximumWidth
            containerView.addSubview(cell)
            cells.append(cell)
        } else {
            cell = cells[index]
        }

        cell.attributedText = attributedText

        if isHeader {
            applyCellHeaderStyling(to: cell)
        } else {
            applyCellNormalStyling(to: cell)
        }

        return cell
    }

    private func calculateCellSize(for cell: LTXLabel, cellPadding: CGFloat) -> CGSize {
        let contentSize = cell.intrinsicContentSize
        return CGSize(
            width: ceil(contentSize.width) + cellPadding * 2,
            height: ceil(contentSize.height) + cellPadding * 2
        )
    }

    private func applyCellHeaderStyling(to cell: LTXLabel) {
        if let attributedText = cell.attributedText.mutableCopy() as? NSMutableAttributedString {
            let range = NSRange(location: 0, length: attributedText.length)

            attributedText.enumerateAttribute(.font, in: range, options: []) {
                value, subRange, _ in
                if let existingFont = value as? UIFont {
                    let boldFont = UIFont.boldSystemFont(ofSize: existingFont.pointSize)
                    attributedText.addAttribute(.font, value: boldFont, range: subRange)
                } else {
                    attributedText.addAttribute(.font, value: theme.fonts.bold, range: subRange)
                }
            }

            cell.attributedText = attributedText
        }
    }

    private func applyCellNormalStyling(to cell: LTXLabel) {
        if let attributedText = cell.attributedText.mutableCopy() as? NSMutableAttributedString {
            let range = NSRange(location: 0, length: attributedText.length)

            attributedText.enumerateAttribute(.foregroundColor, in: range, options: []) {
                value, subRange, _ in
                if value == nil {
                    attributedText.addAttribute(
                        .foregroundColor, value: theme.colors.body, range: subRange
                    )
                }
            }

            cell.attributedText = attributedText
        }
    }

    private func updateCellsAppearance() {
        for (index, cell) in cells.enumerated() {
            let numberOfColumns = widths.count
            let row = index / numberOfColumns
            let isHeaderCell = row == 0

            if isHeaderCell {
                applyCellHeaderStyling(to: cell)
            } else {
                applyCellNormalStyling(to: cell)
            }
        }
    }
}
