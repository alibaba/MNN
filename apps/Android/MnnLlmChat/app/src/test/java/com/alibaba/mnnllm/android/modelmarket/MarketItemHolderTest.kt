package com.alibaba.mnnllm.android.modelmarket

import android.view.LayoutInflater
import android.view.MenuItem
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.PopupMenu
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.DialogUtils
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkObject
import io.mockk.verify
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class MarketItemHolderTest {

    @Test
    fun longPressDelete_pausedModel_confirmDelete_invokesListener() {
        mockkObject(DialogUtils)
        try {
            every { DialogUtils.showDeleteConfirmationDialog(any(), any()) } answers {
                secondArg<() -> Unit>().invoke()
            }

            val activity = Robolectric.buildActivity(AppCompatActivity::class.java).setup().get()
            activity.setTheme(R.style.AppTheme)
            val context = activity
            val itemView = LayoutInflater.from(context)
                .inflate(R.layout.recycle_item_market, FrameLayout(context), false)

            val listener = mockk<ModelMarketItemListener>(relaxed = true)
            val holder = MarketItemHolder(itemView, listener)
            val wrapper = createWrapper(
                modelId = "ModelScope/Test/DeletePath",
                downloadState = DownloadState.PAUSED
            )
            holder.bind(wrapper)

            val popupMenu = PopupMenu(context, itemView.findViewById(R.id.tvStatus))
            popupMenu.menuInflater.inflate(R.menu.market_item_context_menu, popupMenu.menu)
            invokeSetupMenuClickListener(holder, popupMenu, wrapper)
            invokeConfigureMenuVisibility(holder, popupMenu, wrapper.downloadInfo.downloadState)

            assertTrue(popupMenu.menu.findItem(R.id.menu_delete_model).isVisible)
            clickMenuItem(popupMenu, R.id.menu_delete_model)
            verify(exactly = 1) { listener.onDeleteClicked(wrapper) }
        } finally {
            unmockkObject(DialogUtils)
        }
    }

    private fun clickMenuItem(popupMenu: PopupMenu, itemId: Int) {
        val listenerField = PopupMenu::class.java.getDeclaredField("mMenuItemClickListener")
        listenerField.isAccessible = true
        val listener = listenerField.get(popupMenu) as PopupMenu.OnMenuItemClickListener
        val item: MenuItem = popupMenu.menu.findItem(itemId)
        listener.onMenuItemClick(item)
    }

    private fun invokeSetupMenuClickListener(
        holder: MarketItemHolder,
        popupMenu: PopupMenu,
        wrapper: ModelMarketItemWrapper
    ) {
        val method = MarketItemHolder::class.java.getDeclaredMethod(
            "setupMenuClickListener",
            PopupMenu::class.java,
            ModelMarketItemWrapper::class.java,
            ModelMarketItem::class.java
        )
        method.isAccessible = true
        method.invoke(holder, popupMenu, wrapper, wrapper.modelMarketItem)
    }

    private fun invokeConfigureMenuVisibility(
        holder: MarketItemHolder,
        popupMenu: PopupMenu,
        downloadState: Int
    ) {
        val method = MarketItemHolder::class.java.getDeclaredMethod(
            "configureMenuVisibility",
            PopupMenu::class.java,
            Int::class.javaPrimitiveType
        )
        method.isAccessible = true
        method.invoke(holder, popupMenu, downloadState)
    }

    private fun createWrapper(modelId: String, downloadState: Int): ModelMarketItemWrapper {
        val item = ModelMarketItem(
            modelName = "TestModel",
            vendor = "TestVendor",
            sizeB = 1.0,
            tags = emptyList(),
            categories = emptyList(),
            sources = mapOf("ModelScope" to "Test/Repo"),
            modelId = modelId
        )
        return ModelMarketItemWrapper(item, DownloadInfo(downloadState = downloadState))
    }
}
