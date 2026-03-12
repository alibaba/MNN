package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.customview.widget.ViewDragHelper
import kotlin.math.abs

class FullScreenDrawerLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : DrawerLayout(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "FullScreenDrawerLayout"
    }

    private var initialX = 0f
    private var initialY = 0f
    private var leftDragger: ViewDragHelper? = null

    init {
        // Find mLeftDragger to allow manual full-screen swipe capturing
        try {
            val leftDraggerField = DrawerLayout::class.java.getDeclaredField("mLeftDragger")
            leftDraggerField.isAccessible = true
            leftDragger = leftDraggerField.get(this) as? ViewDragHelper
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize reflection for mLeftDragger", e)
        }
    }

    // Minimum distance required to trigger drawer during middle-screen swipe
    private val minSwipeDistanceForDrawer = 80f
    
    override fun onInterceptTouchEvent(ev: MotionEvent): Boolean {
        var intercepted = false
        try {
            intercepted = super.onInterceptTouchEvent(ev)
        } catch (e: Exception) {
            // Ignore native exceptions that sometimes occur in ViewDragHelper
        }

        val x = ev.x
        val y = ev.y

        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                initialX = x
                initialY = y
                // No need to remove mPeekRunnable because without overriding mEdgeSize, 
                // it only triggers on actual edge touches, which is the correct native behavior.
            }
            MotionEvent.ACTION_MOVE -> {
                // If DrawerLayout already decided to intercept (e.g., edge swipe or drawer is open), let it.
                if (intercepted) return true

                val drawerOpen = isDrawerOpen(GravityCompat.START)
                val dx = x - initialX
                val dy = y - initialY
                
                if (!drawerOpen) {
                    // Respect lock mode (e.g. BenchmarkFragment disables drawer)
                    if (getDrawerLockMode(GravityCompat.START) != DrawerLayout.LOCK_MODE_UNLOCKED) {
                        return false
                    }
                    // Only allow full screen swipe for clear horizontal swipes towards the right
                    if (dx > minSwipeDistanceForDrawer && abs(dx) > abs(dy) * 1.5 && initialX < width * 0.8) {
                        try {
                            val drawerView = findDrawerWithGravity(Gravity.LEFT) ?: findDrawerWithGravity(GravityCompat.START)
                            if (drawerView != null) {
                                // Forcefully capture the drawer view to start the drag
                                leftDragger?.captureChildView(drawerView, ev.getPointerId(0))
                                return true
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Reflective capture failed", e)
                        }
                    }
                }
            }
        }
        return intercepted
    }

    private fun findDrawerWithGravity(gravity: Int): View? {
        val absGravity = GravityCompat.getAbsoluteGravity(gravity, layoutDirection)
        for (i in 0 until childCount) {
            val child = getChildAt(i)
            val childAbsGravity = GravityCompat.getAbsoluteGravity((child.layoutParams as LayoutParams).gravity, layoutDirection)
            if ((childAbsGravity and Gravity.HORIZONTAL_GRAVITY_MASK) == (absGravity and Gravity.HORIZONTAL_GRAVITY_MASK)) {
                return child
            }
        }
        return null
    }
}
