package com.alibaba.mnnllm.android.chat.chatlist

import android.content.Context
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import io.noties.markwon.Markwon

class MarkdownMessageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    private var renderedBlocks: List<MarkdownBlockParser.Block> = emptyList()
    private var plainTextMode = false

    init {
        orientation = VERTICAL
    }

    fun showPlainText(text: CharSequence) {
        if (!plainTextMode || childCount != 1 || getChildAt(0) !is TextView) {
            removeAllViews()
            addView(createTextView())
        }
        plainTextMode = true
        renderedBlocks = emptyList()
        (getChildAt(0) as TextView).text = text
    }

    fun renderMarkdown(
        markwon: Markwon,
        sourceText: String,
        preprocessedText: String = sourceText,
        isStreaming: Boolean = false
    ) {
        var blocks = MarkdownBlockParser.parse(sourceText, isStreaming)
        val streamingSuffix = preprocessedText.takeIf { it.startsWith(sourceText) }
            ?.substring(sourceText.length)
            .orEmpty()
        if (streamingSuffix.isNotEmpty()) {
            val lastMarkdownIndex = blocks.indexOfLast { it is MarkdownBlockParser.Block.Markdown }
            blocks = if (lastMarkdownIndex >= 0) {
                blocks.toMutableList().also { updatedBlocks ->
                    val markdownBlock = updatedBlocks[lastMarkdownIndex] as MarkdownBlockParser.Block.Markdown
                    updatedBlocks[lastMarkdownIndex] = markdownBlock.copy(
                        content = markdownBlock.content + streamingSuffix
                    )
                }
            } else {
                blocks + MarkdownBlockParser.Block.Markdown(streamingSuffix)
            }
        }

        plainTextMode = false
        if (!hasSameStructure(renderedBlocks, blocks)) {
            rebuild(markwon, blocks)
            return
        }

        blocks.forEachIndexed { index, block ->
            if (block != renderedBlocks[index]) {
                updateBlock(markwon, getChildAt(index), block)
            }
        }
        renderedBlocks = blocks
    }

    private fun hasSameStructure(
        oldBlocks: List<MarkdownBlockParser.Block>,
        newBlocks: List<MarkdownBlockParser.Block>
    ): Boolean {
        return oldBlocks.size == newBlocks.size && oldBlocks.indices.all { index ->
            oldBlocks[index]::class == newBlocks[index]::class
        }
    }

    private fun rebuild(markwon: Markwon, blocks: List<MarkdownBlockParser.Block>) {
        removeAllViews()
        blocks.forEach { block ->
            val view = when (block) {
                is MarkdownBlockParser.Block.Code -> createCodeBlockView(block.content)
                is MarkdownBlockParser.Block.Markdown,
                is MarkdownBlockParser.Block.Table -> createTextView().also {
                    markwon.setMarkdown(it, block.content)
                }
            }
            addView(view)
        }
        renderedBlocks = blocks
    }

    private fun updateBlock(markwon: Markwon, view: View, block: MarkdownBlockParser.Block) {
        when (block) {
            is MarkdownBlockParser.Block.Code -> {
                val scrollView = view as HorizontalScrollView
                (scrollView.getChildAt(0) as TextView).text = block.content
            }
            is MarkdownBlockParser.Block.Markdown,
            is MarkdownBlockParser.Block.Table -> markwon.setMarkdown(view as TextView, block.content)
        }
    }

    private fun createTextView(): TextView {
        return TextView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
            setTextAppearance(R.style.Light)
            setTextColor(resolveColor(com.google.android.material.R.attr.colorOnSurface))
            setTextSize(TypedValue.COMPLEX_UNIT_PX, resources.getDimension(R.dimen.h3))
            forwardLongClicksToContainer(this)
        }
    }

    private fun createCodeBlockView(code: String): HorizontalScrollView {
        val codeText = TextView(context).apply {
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
            background = ContextCompat.getDrawable(context, R.drawable.bg_markdown_code_block)
            typeface = Typeface.MONOSPACE
            setTextColor(resolveColor(com.google.android.material.R.attr.colorOnSurface))
            setTextSize(TypedValue.COMPLEX_UNIT_PX, resources.getDimension(R.dimen.h4))
            setPadding(dp(12), dp(10), dp(12), dp(10))
            setHorizontallyScrolling(true)
            text = code
            forwardLongClicksToContainer(this)
        }
        return HorizontalScrollView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).also {
                it.topMargin = dp(4)
                it.bottomMargin = dp(4)
            }
            isFillViewport = false
            isHorizontalScrollBarEnabled = true
            isScrollbarFadingEnabled = false
            forwardLongClicksToContainer(this)
            addView(codeText)
        }
    }

    private fun forwardLongClicksToContainer(view: View) {
        view.setOnLongClickListener {
            this@MarkdownMessageView.performLongClick()
        }
    }

    private fun resolveColor(attribute: Int): Int {
        val value = TypedValue()
        context.theme.resolveAttribute(attribute, value, true)
        return if (value.resourceId != 0) {
            ContextCompat.getColor(context, value.resourceId)
        } else {
            value.data
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density + 0.5F).toInt()
    }
}
