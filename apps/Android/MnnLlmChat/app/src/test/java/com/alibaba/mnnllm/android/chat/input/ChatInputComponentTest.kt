package com.alibaba.mnnllm.android.chat.input

import android.app.Application
import android.view.View
import android.view.ViewGroup
import android.widget.HorizontalScrollView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.ContextUsageFormat
import com.alibaba.mnnllm.android.chat.DEFAULT_SANA_PROMPT
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelist.ModelListManager
import io.mockk.every
import io.mockk.mockkObject
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.android.controller.ActivityController
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class, sdk = [28], qualifiers = "en-rUS-mdpi")
class ChatInputComponentTest {
    private val activities = mutableListOf<ActivityController<ChatActivity>>()

    @After
    fun tearDown() {
        activities.forEach { it.destroy() }
        unmockkAll()
    }

    private fun inputUi(): Pair<ChatActivity, ActivityChatBinding> {
        val controller = Robolectric.buildActivity(ChatActivity::class.java).create()
        activities.add(controller)
        val activity = controller.get()
        val content = activity.findViewById<ViewGroup>(android.R.id.content)
        return activity to ActivityChatBinding.bind(content.getChildAt(0))
    }

    private fun layoutInput(binding: ActivityChatBinding, widthDp: Int) {
        val density = binding.root.resources.displayMetrics.density
        val width = (widthDp * density).toInt()
        val height = (640 * density).toInt()
        binding.root.measure(
            View.MeasureSpec.makeMeasureSpec(width, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(height, View.MeasureSpec.EXACTLY)
        )
        binding.root.layout(0, 0, width, height)
    }

    private fun assertCompactInputFits() {
        val (activity, binding) = inputUi()
        binding.btPlus.visibility = View.VISIBLE
        binding.btSwitchAudio.visibility = View.VISIBLE
        binding.btnAgentMode.visibility = View.VISIBLE
        binding.btnToggleThinking.visibility = View.VISIBLE
        binding.tvContextUsage.visibility = View.VISIBLE
        val controls = binding.tvContextUsage.parent as ViewGroup
        val viewport = controls.parent as HorizontalScrollView
        val touchTarget = (48 * activity.resources.displayMetrics.density).toInt()
        for (tokens in listOf(18, 1234)) {
            binding.tvContextUsage.text = activity.getString(
                R.string.context_usage_format,
                ContextUsageFormat.tokens(tokens), ContextUsageFormat.tokens(8192)
            )
            layoutInput(binding, 360)
            assertTrue(controls.width <= viewport.width)
            assertTrue(viewport.left >= binding.btSwitchAudio.right)
            assertTrue(viewport.right <= binding.btnSend.left)
            assertTrue(binding.btnSend.right <= binding.bottomButtonLine.width)
            for (button in listOf(binding.btPlus, binding.btnSend)) {
                assertTrue(button.width >= touchTarget)
                assertTrue(button.height >= touchTarget)
                assertTrue(button.isFocusable)
                assertFalse(button.contentDescription.isNullOrBlank())
            }
            for (button in listOf(binding.btnAgentMode, binding.btnToggleThinking)) {
                assertEquals(1, button.lineCount)
                assertEquals(0, button.layout.getEllipsisCount(0))
            }
        }
        assertEquals(activity.getString(R.string.thinking_mode_label), binding.btnToggleThinking.text.toString())
    }

    @Test
    fun sendButtonAccessibility_tracksGenerationAndLoadingState() {
        mockkObject(ModelListManager, ModelTypeUtils)
        every { ModelListManager.getExtraTags(any()) } returns emptyList()
        every { ModelTypeUtils.isVisualModel(any()) } returns false
        every { ModelTypeUtils.isAudioModel(any()) } returns false
        every { ModelTypeUtils.requiresFaceImageInput(any()) } returns false
        every { ModelTypeUtils.supportAudioOutput(any()) } returns false
        val (activity, binding) = inputUi()
        val component = ChatInputComponent(activity, binding, "test-model", "test-model")
        assertFalse(binding.btnSend.isEnabled)
        assertEquals(activity.getString(R.string.send), binding.btnSend.contentDescription)
        binding.etMessage.setText("Reply only:OK")
        assertTrue(binding.btnSend.isEnabled)
        var stopRequests = 0
        component.setOnStopGenerating { stopRequests++ }
        repeat(2) {
            activity.isGenerating = true
            component.setIsGenerating(true)
            assertEquals(activity.getString(R.string.chat_stop_generation), binding.btnSend.contentDescription)
            assertTrue(binding.btnSend.isEnabled)
            binding.btnSend.performClick()
            assertTrue(activity.isGenerating)
            activity.isLoading = true
            component.onLoadingStatesChanged(true)
            assertTrue(binding.btnSend.isEnabled)
            assertEquals(activity.getString(R.string.chat_stop_generation), binding.btnSend.contentDescription)
            activity.isGenerating = false
            component.setIsGenerating(false)
            assertFalse(binding.btnSend.isEnabled)
            assertEquals(activity.getString(R.string.send), binding.btnSend.contentDescription)
            activity.isLoading = false
            component.onLoadingStatesChanged(false)
            assertTrue(binding.btnSend.isEnabled)
        }
        assertEquals(2, stopRequests)
        binding.etMessage.setText("")
        assertFalse(binding.btnSend.isEnabled)
        assertEquals(activity.getString(R.string.send), binding.btnSend.contentDescription)
    }

    @Test
    @GraphicsMode(GraphicsMode.Mode.NATIVE)
    fun bottomRow_englishFitsAt360dpWithAllControls() {
        assertCompactInputFits()
    }

    @Test
    @Config(qualifiers = "zh-rCN-mdpi")
    @GraphicsMode(GraphicsMode.Mode.NATIVE)
    fun bottomRow_simplifiedChineseFitsAt360dpWithAllControls() {
        assertCompactInputFits()
    }

    @Test
    @Config(qualifiers = "zh-rTW-mdpi")
    @GraphicsMode(GraphicsMode.Mode.NATIVE)
    fun bottomRow_traditionalChineseFitsAt360dpWithAllControls() {
        assertCompactInputFits()
    }

    @Test
    @GraphicsMode(GraphicsMode.Mode.NATIVE)
    fun bottomRow_overflowScrollsWithoutDisplacingSendAt320dp() {
        val (activity, binding) = inputUi()
        binding.btPlus.visibility = View.VISIBLE
        binding.btSwitchAudio.visibility = View.VISIBLE
        binding.btnAgentMode.visibility = View.VISIBLE
        binding.btnToggleThinking.visibility = View.VISIBLE
        binding.tvContextUsage.visibility = View.VISIBLE
        binding.tvContextUsage.text = activity.getString(R.string.context_usage_format, "999.9K", "999.9K")
        binding.btnAgentMode.setText(R.string.agent_mode_chip_confirm)
        layoutInput(binding, 320)
        val controls = binding.tvContextUsage.parent as ViewGroup
        val viewport = controls.parent as HorizontalScrollView
        assertTrue(controls.width > viewport.width)
        assertTrue(viewport.left >= binding.btSwitchAudio.right)
        assertTrue(viewport.right <= binding.btnSend.left)
        assertTrue(binding.btnSend.right <= binding.bottomButtonLine.width)
        val sendLeft = binding.btnSend.left
        viewport.scrollTo(controls.width, 0)
        assertTrue(viewport.scrollX > 0)
        assertEquals(sendLeft, binding.btnSend.left)
        assertEquals((48 * activity.resources.displayMetrics.density).toInt(), binding.btnSend.width)
    }

    @Test
    fun `sana mode should disable send when no image selected even if text exists`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = false,
            inputText = "hello",
            requiresFaceImage = true
        )
        assertFalse(enabled)
    }

    @Test
    fun `sana mode should enable send when image selected`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = true,
            inputText = "",
            requiresFaceImage = true
        )
        assertTrue(enabled)
    }

    @Test
    fun `normal mode should enable send when text exists`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = false,
            inputText = "hello",
            requiresFaceImage = false
        )
        assertTrue(enabled)
    }

    @Test
    fun `sana mode should prefill default prompt when input is blank`() {
        val text = ChatInputComponent.resolveInputTextForModel(
            inputText = "",
            requiresFaceImage = true
        )
        assertEquals(DEFAULT_SANA_PROMPT, text)
    }

    @Test
    fun `sana mode should keep user text when input is not blank`() {
        val text = ChatInputComponent.resolveInputTextForModel(
            inputText = "custom prompt",
            requiresFaceImage = true
        )
        assertEquals("custom prompt", text)
    }
}
