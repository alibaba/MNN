package com.alibaba.mnnllm.android.modelsettings

import android.app.Application
import android.graphics.Rect
import android.os.Looper
import android.view.LayoutInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ListView
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.Insets
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.NestedScrollView
import androidx.lifecycle.Lifecycle
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentSettingsSheetBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import com.alibaba.mnnllm.android.modelmarket.FilterDialogFragment
import com.alibaba.mnnllm.android.modelmarket.FilterState
import com.alibaba.mnnllm.android.modelmarket.Tag
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.android.controller.ActivityController
import org.robolectric.annotation.Config
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class BaseSettingsBottomSheetFragmentTest {

    private lateinit var controller: ActivityController<AppCompatActivity>
    private lateinit var activity: AppCompatActivity

    @Before
    fun setup() {
        controller = Robolectric.buildActivity(AppCompatActivity::class.java)
        activity = controller.get()
        activity.setTheme(R.style.AppTheme)
        controller.setup()
    }

    @After
    fun tearDown() {
        controller.pause().stop().destroy()
    }

    @Test
    fun `resolveConfigFilePath prefers explicit config path`() {
        val resolved = BaseSettingsBottomSheetFragment.resolveConfigFilePath(
            modelId = "Builtin/MNN/test",
            configPath = "/tmp/manual_config.json",
            defaultConfigProvider = { "ignored" }
        )
        assertEquals("/tmp/manual_config.json", resolved)
    }

    @Test
    fun `resolveConfigFilePath returns null when no config file is available`() {
        val resolved = BaseSettingsBottomSheetFragment.resolveConfigFilePath(
            modelId = "",
            configPath = null,
            defaultConfigProvider = { null }
        )
        assertNull(resolved)
    }

    @Test
    @Config(qualifiers = "en-rUS-mdpi")
    fun `filter opens expanded with fixed actions and scrollable overflow after window resize`() {
        val fragment = FilterDialogFragment().apply {
            setAvailableTags((1..40).map { Tag("标签 $it", "Tag $it") })
            setAvailableVendors((1..60).map { "Vendor $it" })
        }
        fragment.showNow(activity.supportFragmentManager, "filter")
        shadowOf(Looper.getMainLooper()).idleFor(1, TimeUnit.SECONDS)
        val sheet = requireNotNull(fragment.requireDialog().findViewById<FrameLayout>(
            com.google.android.material.R.id.design_bottom_sheet
        ))
        val behavior = BottomSheetBehavior.from(sheet)
        assertEquals(ViewGroup.LayoutParams.MATCH_PARENT, sheet.layoutParams.height)
        assertEquals(BottomSheetBehavior.STATE_EXPANDED, behavior.state)
        assertTrue(behavior.skipCollapsed)

        val root = fragment.requireView()
        val scroll = root.findViewById<NestedScrollView>(R.id.filter_scroll_view)
        val header = root.findViewById<View>(R.id.filter_header)
        val actions = root.findViewById<View>(R.id.filter_actions)
        val downloadStates = root.findViewById<ChipGroup>(R.id.download_state_chip_group)
        val decor = fragment.requireDialog().window!!.decorView
        for ((width, height) in listOf(320 to 640, 640 to 320, 280 to 480)) {
            layout(decor, width, height)
            val rootBounds = windowBounds(root)
            val decorBounds = windowBounds(decor)
            assertTrue(decorBounds.contains(rootBounds))
            assertTrue(scroll.height > 0)
            assertTrue(header.bottom <= scroll.top)
            assertTrue(scroll.bottom <= actions.top)
            assertTrue(scroll.getChildAt(0).height > scroll.height)
            assertFilterActionsFit(root)
            val sheetTop = sheet.top
            val actionBounds = windowBounds(actions)
            scroll.scrollTo(0, scroll.getChildAt(0).height)
            assertTrue(scroll.scrollY > 0)
            assertTrue(windowBounds(downloadStates).bottom <= windowBounds(scroll).bottom)
            assertTrue(windowBounds(downloadStates).top >= windowBounds(scroll).top)
            assertEquals(actionBounds, windowBounds(actions))
            assertEquals(sheetTop, sheet.top)
            assertEquals(BottomSheetBehavior.STATE_EXPANDED, behavior.state)
        }
    }

    @Test
    @Config(qualifiers = "en-rUS-mdpi")
    fun `filter accounts for cutouts and system bars without accumulating padding`() {
        val fragment = FilterDialogFragment()
        fragment.showNow(activity.supportFragmentManager, "filter")
        shadowOf(Looper.getMainLooper()).idleFor(1, TimeUnit.SECONDS)
        val root = fragment.requireView()
        val decor = fragment.requireDialog().window!!.decorView
        layout(decor, 640, 320)
        val initialPadding = listOf(root.paddingLeft, root.paddingTop, root.paddingRight, root.paddingBottom)
        val insets = WindowInsetsCompat.Builder()
            .setInsets(WindowInsetsCompat.Type.systemBars(), Insets.of(0, 24, 32, 20))
            .setInsets(WindowInsetsCompat.Type.displayCutout(), Insets.of(40, 0, 0, 0))
            .build()
        ViewCompat.dispatchApplyWindowInsets(root, insets)
        layout(decor, 640, 320)
        val padded = listOf(root.paddingLeft, root.paddingTop, root.paddingRight, root.paddingBottom)
        ViewCompat.dispatchApplyWindowInsets(root, insets)
        layout(decor, 640, 320)
        assertEquals(padded, listOf(root.paddingLeft, root.paddingTop, root.paddingRight, root.paddingBottom))
        val bounds = windowBounds(root)
        val window = windowBounds(decor)
        assertTrue(bounds.left + root.paddingLeft >= window.left + 40)
        assertTrue(bounds.top + root.paddingTop >= window.top + 24)
        assertTrue(bounds.right - root.paddingRight <= window.right - 32)
        assertTrue(bounds.bottom - root.paddingBottom <= window.bottom - 20)
        assertFilterActionsFit(root)

        ViewCompat.dispatchApplyWindowInsets(root, WindowInsetsCompat.Builder().build())
        layout(decor, 640, 320)
        assertEquals(initialPadding, listOf(root.paddingLeft, root.paddingTop, root.paddingRight, root.paddingBottom))
        assertFilterActionsFit(root)
    }

    @Test
    @Config(qualifiers = "zh-rTW")
    fun `filter localizes tags but preserves restored keys live selection clear and confirm`() {
        val expected = FilterState(
            tagKeys = listOf("Vision"),
            vendors = listOf("Vendor A"),
            size = "1B-5B",
            downloadState = DownloadState.DOWNLOAD_SUCCESS
        )
        val updates = mutableListOf<FilterState>()
        val fragment = FilterDialogFragment().apply {
            setAvailableTags(listOf(Tag("unused fallback", "Vision"), Tag("unused fallback", "Chat")))
            setAvailableVendors(listOf("Vendor A", "Vendor B"))
            setCurrentFilterState(expected)
            setOnConfirmListener { updates.add(it) }
        }
        fragment.showNow(activity.supportFragmentManager, "filter")
        shadowOf(Looper.getMainLooper()).idleFor(1, TimeUnit.SECONDS)
        val root = fragment.requireView()
        val tags = root.findViewById<ChipGroup>(R.id.tags_chip_group)
        val vendors = root.findViewById<ChipGroup>(R.id.vendor_chip_group)
        val sizes = root.findViewById<ChipGroup>(R.id.size_chip_group)
        val states = root.findViewById<ChipGroup>(R.id.download_state_chip_group)
        val vision = tags.getChildAt(0) as Chip
        assertEquals(fragment.getString(R.string.model_tag_vision), vision.text.toString())
        assertEquals("Vision", vision.tag)
        assertEquals(expected, updates.last())
        assertTrue(listOf(tags, vendors, sizes, states).all { it.isSingleSelection })

        (tags.getChildAt(1) as Chip).isChecked = true
        (vendors.getChildAt(1) as Chip).isChecked = true
        assertEquals(listOf("Chat"), updates.last().tagKeys)
        assertEquals(listOf("Vendor B"), updates.last().vendors)
        root.findViewById<View>(R.id.clear_button).performClick()
        assertEquals(FilterState(), updates.last())
        assertTrue(listOf(tags, vendors, sizes, states).all { it.checkedChipIds.isEmpty() })
        assertTrue(fragment.requireDialog().isShowing)

        vision.isChecked = true
        (states.getChildAt(2) as Chip).isChecked = true
        val dialog = fragment.requireDialog()
        root.findViewById<View>(R.id.confirm_button).performClick()
        assertEquals(FilterState(tagKeys = listOf("Vision"), downloadState = DownloadState.DOWNLOADING), updates.last())
        assertFalse(dialog.isShowing)
    }

    @Test
    fun `filter accepts empty tag and vendor lists and confirms an empty filter`() {
        var result: FilterState? = null
        val fragment = FilterDialogFragment().apply { setOnConfirmListener { result = it } }
        fragment.showNow(activity.supportFragmentManager, "filter")
        shadowOf(Looper.getMainLooper()).idleFor(1, TimeUnit.SECONDS)
        val root = fragment.requireView()
        assertEquals(0, root.findViewById<ChipGroup>(R.id.tags_chip_group).childCount)
        assertEquals(0, root.findViewById<ChipGroup>(R.id.vendor_chip_group).childCount)
        assertEquals(4, root.findViewById<ChipGroup>(R.id.size_chip_group).childCount)
        assertEquals(3, root.findViewById<ChipGroup>(R.id.download_state_chip_group).childCount)
        root.findViewById<View>(R.id.confirm_button).performClick()
        assertEquals(FilterState(), result)
    }

    @Test
    @Config(qualifiers = "en-rUS-mdpi")
    fun `slider keeps temperature on one line above a full width track on narrow layouts`() {
        val row = SettingsRowSliderSwitchBinding.inflate(LayoutInflater.from(activity), FrameLayout(activity), false)
        row.labelSlider.text = activity.getString(R.string.settings_temperature)
        row.valueSlider.text = "0.85"
        measureRow(row.root, 272)
        assertEquals(1, row.labelSlider.lineCount)
        assertEquals(0, row.labelSlider.layout.getEllipsisCount(0))
        assertSliderRowFits(row)
        val labelWidthWithSwitch = row.labelSlider.width
        row.switchSlider.visibility = View.GONE
        row.labelSlider.text = activity.getString(R.string.settings_ngram_factor)
        measureRow(row.root, 256)
        assertTrue(row.labelSlider.width > labelWidthWithSwitch)
        assertEquals(1, row.labelSlider.lineCount)
        assertEquals(0, row.labelSlider.layout.getEllipsisCount(0))
        assertSliderRowFits(row)
    }

    @Test
    @Config(qualifiers = "en-rUS-mdpi")
    fun `large text slider keeps value and switch separate without splitting labels`() {
        val configuration = android.content.res.Configuration(activity.resources.configuration).apply {
            fontScale = 2f
        }
        val context = androidx.appcompat.view.ContextThemeWrapper(
            activity.createConfigurationContext(configuration), R.style.AppTheme
        )
        val row = SettingsRowSliderSwitchBinding.inflate(LayoutInflater.from(context), FrameLayout(context), false)
        row.labelSlider.text = context.getString(R.string.settings_temperature)
        row.valueSlider.text = "2.00"
        measureRow(row.root, 272)
        assertEquals(1, row.labelSlider.lineCount)
        assertEquals(1, row.valueSlider.lineCount)
        assertTrue(row.labelSlider.width > 0)
        assertSliderRowFits(row)
    }

    @Test
    @Config(qualifiers = "zh-rTW")
    fun `settings localize sampler labels while retaining raw sampler values and enable behavior`() {
        val config = ModelConfig.defaultConfig.deepCopy().apply {
            samplerType = "mixed"
            penaltySampler = "temperature"
            mixedSamplers!!.add("penalty")
        }
        val (fragment, binding) = createSettings(config)
        assertEquals(activity.getString(R.string.settings_temperature), binding.rowMixedTemp.labelSlider.text.toString())
        assertEquals(activity.getString(R.string.settings_penalty), binding.rowMixedPenalty.labelSlider.text.toString())
        assertEquals(activity.getString(R.string.settings_ngram_size), binding.rowMixedNgramSize.labelSlider.text.toString())
        assertEquals(activity.getString(R.string.settings_ngram_factor), binding.rowMixedNgramFactor.labelSlider.text.toString())
        assertEquals("Top K", binding.rowMixedTopK.labelSlider.text.toString())
        assertEquals("Top P", binding.rowMixedTopP.labelSlider.text.toString())
        assertDropDownLabel(binding.dropdownSamplerType, R.string.settings_sampler_type)
        assertDropDownLabel(binding.dropdownMixedPenaltySampler, R.string.settings_penalty_sampler)
        assertDropDownLabel(binding.dropdownPenaltySampler, R.string.settings_penalty_sampler)
        assertDropDownLabel(binding.dropdownBackend, R.string.backend_label)
        assertDropDownLabel(binding.dropdownPrecision, R.string.settings_precision)
        assertEquals(activity.getString(R.string.settings_sampler_mixed), dropDownValue(binding.dropdownSamplerType))
        assertEquals(activity.getString(R.string.settings_temperature), dropDownValue(binding.dropdownMixedPenaltySampler))
        assertEquals("mixed", config.samplerType)
        assertEquals("temperature", config.penaltySampler)

        binding.rowMixedPenalty.switchSlider.isChecked = false
        assertFalse(config.mixedSamplers!!.contains("penalty"))
        assertFalse(binding.rowMixedNgramSize.seekbar.isEnabled)
        assertFalse(binding.rowMixedNgramFactor.seekbar.isEnabled)
        assertFalse(binding.dropdownMixedPenaltySampler.isEnabled)
        binding.rowMixedPenalty.switchSlider.isChecked = true
        assertTrue(config.mixedSamplers!!.contains("penalty"))
        assertTrue(binding.rowMixedNgramSize.seekbar.isEnabled)
        assertTrue(binding.rowMixedNgramFactor.seekbar.isEnabled)
        assertTrue(binding.dropdownMixedPenaltySampler.isEnabled)

        selectDropDownItem(binding.dropdownMixedPenaltySampler, R.string.settings_sampler_greedy)
        assertEquals("greedy", config.penaltySampler)
        selectDropDownItem(binding.dropdownSamplerType, R.string.settings_penalty)
        assertEquals("penalty", config.samplerType)
        assertEquals(View.VISIBLE, binding.containerPenaltySettings.visibility)
        assertEquals(View.GONE, binding.containerMixedSettings.visibility)
        assertEquals(View.GONE, binding.rowPenaltyPenalty.switchSlider.visibility)
        selectDropDownItem(binding.dropdownPenaltySampler, R.string.settings_temperature)
        assertEquals("temperature", config.penaltySampler)
        assertEquals(activity.getString(R.string.settings_penalty), binding.rowPenaltyPenalty.labelSlider.text.toString())
        assertTrue(fragment.isAdded)
    }

    @Test
    fun `slider changes retain existing numeric endpoints for mixed and penalty samplers`() {
        val config = ModelConfig.defaultConfig.deepCopy().apply { samplerType = "mixed" }
        val (_, binding) = createSettings(config)
        assertSliderEndpoints(binding.rowMixedTopK, 1f, 100f) { config.topK!!.toFloat() }
        assertSliderEndpoints(binding.rowMixedTopP, 0f, 1f) { config.topP!! }
        assertSliderEndpoints(binding.rowMixedMinP, 0f, 1f) { config.minP!! }
        assertSliderEndpoints(binding.rowMixedTfsZ, 0f, 1f) { config.tfsZ!! }
        assertSliderEndpoints(binding.rowMixedTypical, 0f, 1f) { config.typical!! }
        assertSliderEndpoints(binding.rowMixedTemp, 0f, 2f) { config.temperature!! }
        assertSliderEndpoints(binding.rowMixedPenalty, 0f, 5f) { config.penalty!! }
        assertSliderEndpoints(binding.rowMixedNgramSize, 1f, 16f) { config.nGram!!.toFloat() }
        assertSliderEndpoints(binding.rowMixedNgramFactor, 1f, 2f) { config.nGramFactor!! }
        selectDropDownItem(binding.dropdownSamplerType, R.string.settings_penalty)
        assertSliderEndpoints(binding.rowPenaltyPenalty, 0f, 5f) { config.penalty!! }
        assertSliderEndpoints(binding.rowPenaltyNgramSize, 1f, 16f) { config.nGram!!.toFloat() }
        assertSliderEndpoints(binding.rowPenaltyNgramFactor, 1f, 2f) { config.nGramFactor!! }
    }

    private fun createSettings(config: ModelConfig): Pair<SettingsBottomSheetFragment, FragmentSettingsSheetBinding> {
        val fragment = SettingsBottomSheetFragment()
        activity.supportFragmentManager.beginTransaction()
            .add(fragment, "settings")
            .setMaxLifecycle(fragment, Lifecycle.State.CREATED)
            .commitNow()
        BaseSettingsBottomSheetFragment::class.java.getDeclaredField("currentConfig").apply {
            isAccessible = true
            set(fragment, config)
        }
        val root = fragment.onCreateView(LayoutInflater.from(activity), FrameLayout(activity), null)
        SettingsBottomSheetFragment::class.java.getDeclaredMethod("updateSamplerSettings").apply {
            isAccessible = true
            invoke(fragment)
        }
        return fragment to FragmentSettingsSheetBinding.bind(root)
    }

    private fun assertSliderEndpoints(
        row: SettingsRowSliderSwitchBinding,
        minimum: Float,
        maximum: Float,
        currentValue: () -> Float
    ) {
        assertEquals(1000, row.seekbar.max)
        val listener = SeekBar::class.java.getDeclaredField("mOnSeekBarChangeListener").apply {
            isAccessible = true
        }.get(row.seekbar) as SeekBar.OnSeekBarChangeListener
        listener.onProgressChanged(row.seekbar, 0, true)
        assertEquals(minimum, currentValue(), 0.0001f)
        listener.onProgressChanged(row.seekbar, 1000, true)
        assertEquals(maximum, currentValue(), 0.0001f)
        listener.onProgressChanged(row.seekbar, 0, false)
        assertEquals(maximum, currentValue(), 0.0001f)
    }

    private fun selectDropDownItem(dropDown: DropDownLineView, labelResource: Int) {
        if (!dropDown.isAttachedToWindow) {
            activity.setContentView(dropDown.rootView)
            controller.visible()
        }
        shadowOf(Looper.getMainLooper()).idle()
        dropDown.requestRectangleOnScreen(Rect(0, 0, dropDown.width, dropDown.height), true)
        assertTrue(dropDown.isShown)
        assertTrue(dropDown.performClick())
        shadowOf(Looper.getMainLooper()).idle()

        val popup = requireNotNull(shadowOf(activity.application).latestPopupWindow)
        assertTrue(popup.isShowing)
        val menuList = popup.contentView as ListView
        val adapter = requireNotNull(menuList.adapter)
        val label = activity.getString(labelResource)
        val position = (0 until adapter.count).single {
            (adapter.getItem(it) as MenuItem).title.toString() == label
        }
        val itemView = adapter.getView(position, null, menuList)
        assertEquals(label, itemView.findViewById<TextView>(androidx.appcompat.R.id.title).text.toString())
        assertTrue(menuList.performItemClick(itemView, position, adapter.getItemId(position)))
        assertEquals(label, dropDownValue(dropDown))
        assertFalse(popup.isShowing)
    }

    private fun assertDropDownLabel(dropDown: DropDownLineView, resource: Int) {
        assertEquals(activity.getString(resource), dropDown.findViewById<TextView>(R.id.tv_label_sampler_type).text.toString())
    }

    private fun dropDownValue(dropDown: DropDownLineView): String =
        dropDown.findViewById<TextView>(R.id.tv_sampler_type_value).text.toString()

    private fun assertSliderRowFits(row: SettingsRowSliderSwitchBinding) {
        assertTrue(row.seekbar.top >= row.labelValueContainer.bottom)
        assertEquals(row.root.paddingLeft, row.seekbar.left)
        assertEquals(row.root.width - row.root.paddingRight, row.seekbar.right)
        assertTrue(row.labelSlider.right <= row.valueSlider.left)
        if (row.switchSlider.visibility == View.VISIBLE) {
            assertTrue(row.valueSlider.right <= row.switchSlider.left)
        }
        assertTrue(row.valueSlider.right <= row.labelValueContainer.width)
    }

    private fun assertFilterActionsFit(root: View) {
        val bounds = windowBounds(root)
        for (id in listOf(R.id.clear_button, R.id.confirm_button)) {
            val button = root.findViewById<View>(id)
            val buttonBounds = windowBounds(button)
            assertTrue(button.width > 0 && button.height > 0)
            assertTrue(buttonBounds.left >= bounds.left + root.paddingLeft)
            assertTrue(buttonBounds.right <= bounds.right - root.paddingRight)
            assertTrue(buttonBounds.bottom <= bounds.bottom - root.paddingBottom)
        }
    }

    private fun windowBounds(view: View): Rect {
        val position = IntArray(2)
        view.getLocationInWindow(position)
        return Rect(position[0], position[1], position[0] + view.width, position[1] + view.height)
    }

    private fun measureRow(view: View, width: Int) {
        view.measure(
            View.MeasureSpec.makeMeasureSpec(width, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )
        view.layout(0, 0, width, view.measuredHeight)
    }

    private fun layout(view: View, width: Int, height: Int) {
        view.measure(
            View.MeasureSpec.makeMeasureSpec(width, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(height, View.MeasureSpec.EXACTLY)
        )
        view.layout(0, 0, width, height)
    }
}

