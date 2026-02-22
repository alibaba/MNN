package com.alibaba.mnnllm.android.hotspot

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.util.TypedValue
import androidx.appcompat.widget.AppCompatTextView

/**
 * A TextView that automatically cycles through a list of (text, durationMs) pairs.
 * Font size is auto-shrunk to fit within one line if the text is too long.
 */
class LabelCyclerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : AppCompatTextView(context, attrs, defStyleAttr) {

    private val handler = Handler(Looper.getMainLooper())
    private var entries: List<Pair<String, Long>> = emptyList()
    private var currentIndex = 0
    private val defaultSizeSp = 17f   // base font size (slightly bigger than before)
    private val minSizeSp = 9f

    private val advanceRunnable = object : Runnable {
        override fun run() {
            if (entries.isEmpty()) return
            currentIndex = (currentIndex + 1) % entries.size
            showCurrent()
        }
    }

    fun setEntries(list: List<Pair<String, Long>>) {
        handler.removeCallbacks(advanceRunnable)
        entries = list
        currentIndex = 0
        if (list.isNotEmpty()) showCurrent()
    }

    private fun showCurrent() {
        val (label, duration) = entries[currentIndex]
        // Measure text width; shrink font until it fits in one line
        var sp = defaultSizeSp
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        paint.textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, sp, resources.displayMetrics)
        val availableWidth = if (width > 0) width.toFloat() else resources.displayMetrics.widthPixels * 0.85f
        while (sp > minSizeSp && paint.measureText(label) > availableWidth) {
            sp -= 0.5f
            paint.textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, sp, resources.displayMetrics)
        }
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        text = label
        handler.postDelayed(advanceRunnable, duration)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        handler.removeCallbacks(advanceRunnable)
    }
}