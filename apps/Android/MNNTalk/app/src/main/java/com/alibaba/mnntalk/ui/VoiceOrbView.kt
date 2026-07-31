package com.alibaba.mnntalk.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.Shader
import android.graphics.SweepGradient
import android.util.AttributeSet
import android.view.View
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class VoiceOrbView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    enum class Mode {
        IDLE,
        PREPARING,
        LISTENING,
        THINKING,
        SPEAKING,
        ERROR
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val auraPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val veilPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = resources.displayMetrics.density
        alpha = 110
    }

    private var mode = Mode.PREPARING
    private val blobPath = Path()
    private val auraPath = Path()
    private var colorGradient: SweepGradient? = null
    private var lightGradient: RadialGradient? = null
    private val gradientMatrix = Matrix()
    private var startNanos = System.nanoTime()
    private var attached = false

    fun setMode(newMode: Mode) {
        if (mode == newMode) return
        mode = newMode
        startNanos = System.nanoTime()
        invalidate()
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        attached = true
        if (isAnimatedMode()) {
            postInvalidateOnAnimation()
        }
    }

    override fun onDetachedFromWindow() {
        attached = false
        super.onDetachedFromWindow()
    }

    override fun onSizeChanged(width: Int, height: Int, oldWidth: Int, oldHeight: Int) {
        super.onSizeChanged(width, height, oldWidth, oldHeight)
        val centerX = width / 2f
        val centerY = height / 2f
        val radius = minOf(width, height) * BASE_RADIUS
        colorGradient = SweepGradient(
            centerX,
            centerY,
            intArrayOf(PINK, LILAC, BLUE, CYAN, MINT, WARM, PINK),
            floatArrayOf(0f, 0.18f, 0.38f, 0.57f, 0.72f, 0.88f, 1f)
        )
        lightGradient = RadialGradient(
            centerX - radius * 0.30f,
            centerY - radius * 0.34f,
            radius * 1.35f,
            intArrayOf(0xCCFFFFFF.toInt(), 0x55FFFFFF, 0x00FFFFFF),
            floatArrayOf(0f, 0.46f, 1f),
            Shader.TileMode.CLAMP
        )
        fillPaint.shader = colorGradient
        auraPaint.shader = colorGradient
        veilPaint.shader = lightGradient
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val elapsed = (System.nanoTime() - startNanos) / 1_000_000_000f
        val centerX = width / 2f
        val centerY = height / 2f
        val baseRadius = minOf(width, height) * BASE_RADIUS
        val radius = baseRadius * pulseScale(elapsed)
        val waveAmount = waveAmountForMode()

        gradientMatrix.setRotate(
            elapsed * rotationSpeedForMode(),
            centerX,
            centerY
        )
        colorGradient?.setLocalMatrix(gradientMatrix)

        buildBlobPath(
            path = auraPath,
            centerX = centerX,
            centerY = centerY,
            radius = radius * 1.15f,
            elapsed = elapsed,
            amount = waveAmount * 0.70f
        )
        auraPaint.alpha = if (isAnimatedMode()) 22 else 14
        canvas.drawPath(auraPath, auraPaint)

        buildBlobPath(
            path = blobPath,
            centerX = centerX,
            centerY = centerY,
            radius = radius,
            elapsed = elapsed,
            amount = waveAmount
        )
        fillPaint.alpha = if (mode == Mode.ERROR) 185 else 235
        canvas.drawPath(blobPath, fillPaint)
        canvas.drawPath(blobPath, veilPaint)
        canvas.drawPath(blobPath, outlinePaint)

        if (attached && isAnimatedMode()) {
            postInvalidateOnAnimation()
        }
    }

    private fun buildBlobPath(
        path: Path,
        centerX: Float,
        centerY: Float,
        radius: Float,
        elapsed: Float,
        amount: Float
    ) {
        path.reset()
        for (index in 0..POINT_COUNT) {
            val angle = index.toFloat() / POINT_COUNT * TWO_PI
            val distortion = amount * (
                0.52f * sin(angle * 3f + elapsed * 2.0f) +
                    0.30f * sin(angle * 5f - elapsed * 1.55f) +
                    0.18f * sin(angle * 7f + elapsed * 2.45f)
                )
            val currentRadius = radius * (1f + distortion)
            val x = centerX + cos(angle) * currentRadius
            val y = centerY + sin(angle) * currentRadius
            if (index == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }
        path.close()
    }

    private fun pulseScale(elapsed: Float): Float {
        val amount = when (mode) {
            Mode.LISTENING -> 0.025f
            Mode.THINKING -> 0.017f
            Mode.SPEAKING -> 0.035f
            Mode.PREPARING -> 0.010f
            Mode.ERROR,
            Mode.IDLE -> 0f
        }
        return 1f + amount * sin(elapsed * TWO_PI * 0.72f)
    }

    private fun waveAmountForMode(): Float {
        return when (mode) {
            Mode.SPEAKING -> 0.070f
            Mode.LISTENING -> 0.052f
            Mode.THINKING -> 0.035f
            Mode.PREPARING -> 0.014f
            Mode.ERROR -> 0.008f
            Mode.IDLE -> 0f
        }
    }

    private fun rotationSpeedForMode(): Float {
        return when (mode) {
            Mode.SPEAKING -> 15f
            Mode.LISTENING -> 11f
            Mode.THINKING -> 8f
            Mode.PREPARING -> 4f
            Mode.ERROR,
            Mode.IDLE -> 0f
        }
    }

    private fun isAnimatedMode(): Boolean {
        return mode == Mode.PREPARING ||
            mode == Mode.LISTENING ||
            mode == Mode.THINKING ||
            mode == Mode.SPEAKING
    }

    private companion object {
        const val POINT_COUNT = 96
        const val BASE_RADIUS = 0.335f
        const val TWO_PI = (2.0 * PI).toFloat()
        const val PINK = 0xFFFF91C8.toInt()
        const val LILAC = 0xFFB39AFF.toInt()
        const val BLUE = 0xFF769AFF.toInt()
        const val CYAN = 0xFF75E5E3.toInt()
        const val MINT = 0xFFA4E8BC.toInt()
        const val WARM = 0xFFFFD6A1.toInt()
    }
}
