package com.alibaba.mnnllm.android.widgets

// PieProgressView.kt
// ProgressPieView.kt
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.annotation.ColorInt
import androidx.annotation.FloatRange
import com.alibaba.mnnllm.android.R

class ProgressPieView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // 画扇形的画笔
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    // 画外部圆环的画笔
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    private val rectF = RectF()

    // 属性
    private var progress = 0f
    private var startAngle = -90f

    init {
        val typedArray = context.obtainStyledAttributes(
            attrs,
            R.styleable.ProgressPieView,
            defStyleAttr,
            R.style.Widget_App_ProgressPieView
        )

        val fillColor = typedArray.getColor(
            R.styleable.ProgressPieView_pie_fillColor,
            Color.BLACK // 备用颜色
        )
        val ringColor = typedArray.getColor(
            R.styleable.ProgressPieView_pie_ringColor,
            Color.BLACK // 备用颜色
        )
        val ringWidth = typedArray.getDimension(
            R.styleable.ProgressPieView_pie_ringWidth,
            0f
        )

        progress = typedArray.getFloat(R.styleable.ProgressPieView_pie_progress, progress)
        startAngle = typedArray.getFloat(R.styleable.ProgressPieView_pie_startAngle, startAngle)
        typedArray.recycle()

        // 配置画扇形的画笔
        fillPaint.style = Paint.Style.FILL
        fillPaint.color = fillColor

        // 配置画圆环的画笔
        ringPaint.style = Paint.Style.STROKE
        ringPaint.color = ringColor
        ringPaint.strokeWidth = ringWidth
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val sweepAngle = (progress / 100f) * 360f
        if (sweepAngle <= 0 && ringPaint.strokeWidth <= 0) {
            return // 如果没进度也没圆环，就不画了
        }

        // 计算中心和半径
        val centerX = width / 2f
        val centerY = height / 2f

        // 半径需要为外部圆环的线宽留出空间
        val halfRingWidth = ringPaint.strokeWidth / 2f
        val radius = width.coerceAtMost(height) / 2f - halfRingWidth

        // 1. 绘制外部的完整圆环
        if (ringPaint.strokeWidth > 0) {
            canvas.drawCircle(centerX, centerY, radius, ringPaint)
        }

        // 2. 绘制内部的实心扇形
        if (sweepAngle > 0) {
            // 扇形的绘制区域要稍微向内收缩，以免覆盖圆环
            val inset = ringPaint.strokeWidth
            rectF.set(
                paddingLeft + inset,
                paddingTop + inset,
                width - paddingRight - inset,
                height - paddingBottom - inset
            )
            canvas.drawArc(rectF, startAngle, sweepAngle, true, fillPaint)
        }
    }

    /**
     * 以编程方式设置进度
     */
    fun setProgress(@FloatRange(from = 0.0, to = 100.0) value: Float) {
        this.progress = value.coerceIn(0f, 100f)
        invalidate() // 请求重绘
    }

    /**
     * 获取当前进度
     */
    fun getProgress(): Float = progress
}