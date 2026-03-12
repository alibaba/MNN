package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mnnllm.android.R
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class PopupWindowHelperTest {

    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test
    fun createPopupView_buildsThreeRowsWithTrailingIcons() {
        val popupView = PopupWindowHelper.createPopupView(
            LayoutInflater.from(context),
            null
        )

        val container = popupView.findViewById<LinearLayout>(R.id.popup_menu_root)
        assertNotNull(container)
        assertEquals(3, container.childCount)

        assertNotNull(popupView.findViewById<TextView>(R.id.assistant_text_copy))
        assertNotNull(popupView.findViewById<TextView>(R.id.assistant_text_select))
        assertNotNull(popupView.findViewById<TextView>(R.id.assistant_text_report))

        assertNotNull(popupView.findViewById<ImageView>(R.id.assistant_text_copy_icon))
        assertNotNull(popupView.findViewById<ImageView>(R.id.assistant_text_select_icon))
        assertNotNull(popupView.findViewById<ImageView>(R.id.assistant_text_report_icon))
    }

    @Test
    fun calculatePopupPosition_prefersBelowAnchorWhenSpaceAllows() {
        val position = PopupWindowHelper.calculatePopupPosition(
            anchorBounds = PopupWindowHelper.AnchorBounds(
                left = 100,
                top = 120,
                right = 300,
                bottom = 200
            ),
            popupWidth = 180,
            popupHeight = 160,
            screenWidth = 480,
            screenHeight = 800,
            margin = 12,
            spacing = 8
        )

        assertEquals(100, position.x)
        assertEquals(208, position.y)
    }

    @Test
    fun calculatePopupPosition_flipsAboveWhenBelowSpaceIsInsufficient() {
        val position = PopupWindowHelper.calculatePopupPosition(
            anchorBounds = PopupWindowHelper.AnchorBounds(
                left = 220,
                top = 620,
                right = 420,
                bottom = 700
            ),
            popupWidth = 180,
            popupHeight = 160,
            screenWidth = 480,
            screenHeight = 800,
            margin = 12,
            spacing = 8
        )

        assertEquals(220, position.x)
        assertEquals(452, position.y)
    }

    @Test
    fun calculatePopupPosition_clampsWithinScreenMargins() {
        val position = PopupWindowHelper.calculatePopupPosition(
            anchorBounds = PopupWindowHelper.AnchorBounds(
                left = 360,
                top = 120,
                right = 470,
                bottom = 180
            ),
            popupWidth = 180,
            popupHeight = 160,
            screenWidth = 480,
            screenHeight = 800,
            margin = 12,
            spacing = 8
        )

        assertEquals(288, position.x)
        assertEquals(188, position.y)
    }
}
