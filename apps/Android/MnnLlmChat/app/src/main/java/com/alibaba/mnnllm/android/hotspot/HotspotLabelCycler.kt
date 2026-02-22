package com.alibaba.mnnllm.android.hotspot

import android.content.Context
import com.alibaba.mnnllm.android.R

/**
 * Builds the list of (label, durationMs) pairs consumed by [LabelCyclerView].
 *
 * Rules:
 *  - Each language slot lasts NORMAL_MS (700 ms).
 *  - English already appears every 5th slot (indices 0, 5, 10, … in the arrays)
 *    and stays for ENGLISH_MS (1 400 ms = 2× normal).
 *
 * The string-arrays qr_wifi_labels / qr_url_labels were constructed so that
 * every 5th entry (0-based) is already the English string.
 */
object HotspotLabelCycler {

    private const val NORMAL_MS = 700L
    private const val ENGLISH_MS = 1_400L   // 2× normal

    fun buildWifiEntries(context: Context): List<Pair<String, Long>> =
        build(context, R.array.qr_wifi_labels)

    fun buildUrlEntries(context: Context): List<Pair<String, Long>> =
        build(context, R.array.qr_url_labels)

    private fun build(context: Context, arrayRes: Int): List<Pair<String, Long>> {
        val labels = context.resources.getStringArray(arrayRes)
        return labels.mapIndexed { index, label ->
            val duration = if (index % 5 == 0) ENGLISH_MS else NORMAL_MS
            label to duration
        }
    }
}