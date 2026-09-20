package com.alibaba.mnnllm.android.chat.chatlist

import android.content.Context
import android.graphics.Typeface
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.UiUtils.dpToPx
import com.alibaba.mnnllm.android.utils.UiUtils.getThemeColor
import io.noties.markwon.Markwon

/**
 * Renders a GFM table with one View per row.
 *
 * Markwon draws a whole table as TableRowSpans inside a single TextView, so appending a row
 * rebuilds every span and needs two layout passes (column widths, then row heights). The
 * intermediate pass gets drawn, which makes the table reflow visibly on every appended row while
 * streaming.
 */
internal class MarkdownTableView(
    context: Context,
    private val markwon: Markwon
) : LinearLayout(context) {

    companion object {
        private const val MIN_CELL_WIDTH_DP = 96
        private const val CELL_PADDING_H_DP = 8
        private const val CELL_PADDING_V_DP = 6
        /** Horizontal space taken by the avatar and bubble insets, used before the first layout. */
        private const val ESTIMATED_HORIZONTAL_INSET_DP = 72
        private const val GRID_LINE_PX = 1
        /** Characters that make a cell worth handing to Markwon instead of setting it verbatim. */
        private const val MARKDOWN_CHARS = "*_`~[]()$<>\\&"
    }

    private val gridLineColor = context.getThemeColor(
        com.google.android.material.R.attr.colorOutlineVariant
    )
    private val headerBackgroundColor = context.getThemeColor(
        com.google.android.material.R.attr.colorSurfaceVariant
    )
    private val cellBackgroundColor = context.getThemeColor(
        com.google.android.material.R.attr.colorSurface
    )
    private val cellTextColor = context.getThemeColor(
        com.google.android.material.R.attr.colorOnSurface
    )
    private val cellTextSizePx = resources.getDimension(R.dimen.h4)

    private var renderedRows: List<List<String>> = emptyList()
    private var columnCount = 0
    private var cellWidthPx = 0

    init {
        orientation = VERTICAL
        // Rows leave 1px gaps, so this colour shows through as the grid lines.
        setBackgroundColor(gridLineColor)
    }

    fun update(content: String) {
        val rows = MarkdownBlockParser.parseTableRows(content)
        if (rows.isEmpty()) {
            removeAllViews()
            renderedRows = emptyList()
            columnCount = 0
            return
        }

        val columns = rows[0].size
        if (columnCount != columns || !isAppendOnly(renderedRows, rows)) {
            removeAllViews()
            renderedRows = emptyList()
            columnCount = columns
            cellWidthPx = cellWidthFor(columns, estimatedViewportWidth())
        }

        for (index in renderedRows.size until rows.size) {
            addView(createRow(rows[index], isHeader = index == 0))
        }
        renderedRows = rows
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // A HorizontalScrollView measures its child with UNSPECIFIED mode but still reports the
        // viewport width as the spec size, which is the only reliable way to learn it: this view
        // is as wide as its own content, so reading its width would feed the old value back in.
        val viewport = MeasureSpec.getSize(widthMeasureSpec)
        if (viewport > 0 && columnCount > 0) {
            applyCellWidth(cellWidthFor(columnCount, viewport))
        }
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
    }

    /**
     * Resizes the cells that are already on screen. Rebuilding them instead would throw away the
     * whole table on the first measure pass, which is exactly the flicker this view exists to
     * avoid.
     */
    private fun applyCellWidth(target: Int) {
        if (target == cellWidthPx) return
        cellWidthPx = target
        for (rowIndex in 0 until childCount) {
            val row = getChildAt(rowIndex) as? LinearLayout ?: continue
            for (cellIndex in 0 until row.childCount) {
                row.getChildAt(cellIndex).layoutParams.width = target
            }
        }
    }

    /** Appending is only safe while the rows already on screen stay an unchanged prefix. */
    private fun isAppendOnly(old: List<List<String>>, new: List<List<String>>): Boolean {
        if (old.size > new.size) return false
        return old.indices.all { old[it] == new[it] }
    }

    private fun cellWidthFor(columnCount: Int, viewport: Int): Int {
        if (columnCount <= 0) return context.dpToPx(MIN_CELL_WIDTH_DP)
        // Narrow tables fill the width; wide ones keep a readable column width and scroll.
        return maxOf(context.dpToPx(MIN_CELL_WIDTH_DP), viewport / columnCount)
    }

    /** Only used before the first measure pass, when the real viewport width is unknown. */
    private fun estimatedViewportWidth(): Int {
        return resources.displayMetrics.widthPixels - context.dpToPx(ESTIMATED_HORIZONTAL_INSET_DP)
    }

    private fun createRow(cells: List<String>, isHeader: Boolean): LinearLayout {
        val background = if (isHeader) headerBackgroundColor else cellBackgroundColor
        return LinearLayout(context).apply {
            orientation = HORIZONTAL
            isBaselineAligned = false
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
                .also { it.bottomMargin = GRID_LINE_PX }
            cells.forEachIndexed { index, cell ->
                addView(createCell(cell, isHeader, background, isLast = index == cells.lastIndex))
            }
        }
    }

    private fun createCell(
        cell: String,
        isHeader: Boolean,
        background: Int,
        isLast: Boolean
    ): TextView {
        return TextView(context).apply {
            layoutParams = LayoutParams(cellWidthPx, LayoutParams.MATCH_PARENT).also {
                if (!isLast) it.rightMargin = GRID_LINE_PX
            }
            setBackgroundColor(background)
            setPadding(
                context.dpToPx(CELL_PADDING_H_DP),
                context.dpToPx(CELL_PADDING_V_DP),
                context.dpToPx(CELL_PADDING_H_DP),
                context.dpToPx(CELL_PADDING_V_DP)
            )
            gravity = Gravity.CENTER_VERTICAL
            setTextColor(cellTextColor)
            setTextSize(TypedValue.COMPLEX_UNIT_PX, cellTextSizePx)
            if (isHeader) {
                setTypeface(typeface, Typeface.BOLD)
            }
            if (cell.none { it in MARKDOWN_CHARS }) {
                text = cell
            } else {
                markwon.setMarkdown(this, cell)
            }
            forwardLongClickToContainer(this)
        }
    }

    /**
     * Long clicks do not bubble up on their own, and Markwon installs a movement method on cells
     * that would otherwise swallow them, so the chat's copy / report menu would never open on a
     * table. Hand the event to the HorizontalScrollView wrapper that owns the real handler.
     */
    private fun forwardLongClickToContainer(view: View) {
        view.setOnLongClickListener {
            (parent as? View)?.performLongClick() ?: false
        }
    }
}
