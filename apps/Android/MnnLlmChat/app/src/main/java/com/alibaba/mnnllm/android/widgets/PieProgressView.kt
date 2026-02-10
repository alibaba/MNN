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

    //Paint for drawing sectors
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    //Paint for drawing outer ring
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    private val rectF = RectF()

    //Properties
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
            Color.BLACK //Fallback color
        )
        val ringColor = typedArray.getColor(
            R.styleable.ProgressPieView_pie_ringColor,
            Color.BLACK //Fallback color
        )
        val ringWidth = typedArray.getDimension(
            R.styleable.ProgressPieView_pie_ringWidth,
            0f
        )

        progress = typedArray.getFloat(R.styleable.ProgressPieView_pie_progress, progress)
        startAngle = typedArray.getFloat(R.styleable.ProgressPieView_pie_startAngle, startAngle)
        typedArray.recycle()

        //Configure paint for drawing sectors
        fillPaint.style = Paint.Style.FILL
        fillPaint.color = fillColor

        //Configure paint for drawing ring
        ringPaint.style = Paint.Style.STROKE
        ringPaint.color = ringColor
        ringPaint.strokeWidth = ringWidth
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val sweepAngle = (progress / 100f) * 360f
        if (sweepAngle <= 0 && ringPaint.strokeWidth <= 0) {
            return //Don't draw if there's no progress and no ring
        }

        //Calculate center and radius
        val centerX = width / 2f
        val centerY = height / 2f

        //Radius needs to leave space for outer ring line width
        val halfRingWidth = ringPaint.strokeWidth / 2f
        val radius = width.coerceAtMost(height) / 2f - halfRingWidth

        //1. Draw complete outer ring
        if (ringPaint.strokeWidth > 0) {
            canvas.drawCircle(centerX, centerY, radius, ringPaint)
        }

        //2. Draw solid inner sector
        if (sweepAngle > 0) {
            //Sector drawing area should shrink inward slightly to avoid covering the ring
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

    /** * programmaticallysettingprogress*/
    fun setProgress(@FloatRange(from = 0.0, to = 100.0) value: Float) {
        this.progress = value.coerceIn(0f, 100f)
        invalidate() //Request redraw
    }

    /** * getcurrentprogress*/
    fun getProgress(): Float = progress
}