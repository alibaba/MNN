package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.drawable.GradientDrawable
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.LinearLayout
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.ViewPerformanceMetricBinding

/**
 * Custom view component for displaying performance metrics
 * Similar to iOS PerformanceMetricView with icon, title, value, and subtitle
 */
class PerformanceMetricView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    private val binding: ViewPerformanceMetricBinding
    
    init {
        binding = ViewPerformanceMetricBinding.inflate(LayoutInflater.from(context), this, true)
        orientation = VERTICAL
        
        // Set default styling
        val padding = resources.getDimensionPixelSize(R.dimen.performance_metric_padding)
        setPadding(padding, padding, padding, padding)
    }
    
    /**
     * Set performance metric data
     * @param icon Resource ID for the icon
     * @param title Main title text
     * @param value Primary value to display (e.g., "121.13 t/s")
     * @param subtitle Secondary text (e.g., "Prompt Processing")
     * @param colorResId Color resource ID for theming
     * @param stdDev Standard deviation value to display next to the value (e.g., "±3.88")
     */
    fun setMetricData(
        icon: Int,
        title: String,
        value: String,
        subtitle: String,
        colorResId: Int,
        stdDev: String? = null
    ) {
        val color = ContextCompat.getColor(context, colorResId)
        
        // Set icon with circular background
        binding.metricIcon.setImageResource(icon)
        setIconBackground(color)
        
        // Set texts
        binding.metricTitle.text = title
        binding.metricValue.text = value
        binding.metricSubtitle.text = subtitle
        
        // Handle standard deviation display
        if (stdDev != null && stdDev.isNotEmpty()) {
            binding.metricStdDev.text = stdDev
            binding.metricStdDev.setTextColor(color)
            binding.metricStdDev.visibility = android.view.View.VISIBLE
        } else {
            binding.metricStdDev.visibility = android.view.View.GONE
        }
        
        // Apply color theming
        binding.metricValue.setTextColor(color)
        binding.metricIcon.imageTintList = ColorStateList.valueOf(color)
    }
    
    /**
     * Create circular gradient background for icon
     */
    private fun setIconBackground(color: Int) {
        val gradientDrawable = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            
            // Create gradient colors with transparency
            val startColor = (color and 0x00FFFFFF) or 0x33000000 // 20% opacity
            val endColor = (color and 0x00FFFFFF) or 0x1A000000   // 10% opacity
            
            colors = intArrayOf(startColor, endColor)
            gradientType = GradientDrawable.RADIAL_GRADIENT
            gradientRadius = 50f
        }
        
        binding.iconBackground.background = gradientDrawable
    }
    
    /**
     * Convenience method for speed statistics with string resource support
     */
    fun setSpeedMetric(
        icon: Int,
        titleResId: Int,
        stats: SpeedStatistics?,
        colorResId: Int
    ) {
        val title = context.getString(titleResId)
        if (stats != null) {
            val value = "%.1f t/s".format(stats.average)
            val stdDev = "±%.1f".format(stats.stdev)
            setMetricData(icon, title, value, context.getString(R.string.prefill_speed_subtitle), colorResId, stdDev)
        } else {
            setMetricData(icon, title, context.getString(R.string.not_available), context.getString(R.string.prefill_speed_subtitle), colorResId)
        }
    }
    
    /**
     * Convenience method for memory metric
     */
    fun setMemoryMetric(
        maxMemoryKb: Long,
        totalMemoryKb: Long,
        colorResId: Int
    ) {
        // Format memory values with appropriate units (MB or GB)
        val maxMemoryFormatted = formatMemorySize(maxMemoryKb)
        val totalMemoryFormatted = formatMemorySize(totalMemoryKb)
        
        setMetricData(
            R.drawable.ic_memorychip,
            context.getString(R.string.memory_usage_title),
            maxMemoryFormatted,
            context.getString(R.string.memory_usage_subtitle),
            colorResId
        )
    }
    
    /**
     * Format memory size with appropriate unit (MB or GB)
     */
    private fun formatMemorySize(memoryKb: Long): String {
        val memoryMB = memoryKb / 1024.0
        return if (memoryMB >= 1024.0) {
            val memoryGB = memoryMB / 1024.0
            "%.1f GB".format(memoryGB)
        } else {
            "%.0f MB".format(memoryMB)
        }
    }
    
    /**
     * Update metric with simple title, value, and icon name
     * @param title Metric title
     * @param value Metric value
     * @param iconName Icon resource name (e.g., "ic_clock")
     */
    fun updateMetric(title: String, value: String, iconName: String) {
        val iconResId = getIconResourceId(iconName)
        val colorResId = R.color.benchmark_accent
        
        setMetricData(
            icon = iconResId,
            title = title,
            value = value,
            subtitle = title,
            colorResId = colorResId
        )
    }
    
    /**
     * Get icon resource ID from resource name
     */
    private fun getIconResourceId(iconName: String): Int {
        return try {
            val resourceName = if (iconName.startsWith("ic_")) iconName else "ic_$iconName"
            val resourceId = resources.getIdentifier(resourceName, "drawable", context.packageName)
            if (resourceId != 0) resourceId else R.drawable.ic_clock
        } catch (e: Exception) {
            R.drawable.ic_clock
        }
    }
    
    /**
     * Convenience method for total time metric
     */
    fun setTotalTimeMetric(
        totalTimeSeconds: Double,
        colorResId: Int
    ) {
        val formattedTime = formatTime(totalTimeSeconds)
        setMetricData(
            R.drawable.ic_clock,
            context.getString(R.string.total_tokens_title),
            formattedTime,
            context.getString(R.string.total_tokens_subtitle),
            colorResId
        )
    }
    
    /**
     * Format time in seconds to appropriate unit (ms, s, or min)
     */
    private fun formatTime(seconds: Double): String {
        return when {
            seconds < 1.0 -> "%.0f ms".format(seconds * 1000)
            seconds < 60.0 -> "%.3f s".format(seconds)
            else -> "%.1f min".format(seconds / 60.0)
        }
    }
    
    /**
     * Convenience method for total tokens metric (deprecated, use setTotalTimeMetric instead)
     */
    fun setTotalTokensMetric(
        totalTokens: Int,
        colorResId: Int
    ) {
        setMetricData(
            R.drawable.ic_clock,
            context.getString(R.string.total_tokens_title),
            "$totalTokens",
            context.getString(R.string.total_tokens_subtitle),
            colorResId
        )
    }
}
