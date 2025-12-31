package com.alibaba.mnnllm.android.debug

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.motion.widget.MotionLayout
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelmarket.SourceProvider
import com.alibaba.mnnllm.android.modelmarket.SourceSelectionProvider

class DebugAnimationActivity : AppCompatActivity() {

    private lateinit var motionLayout: MotionLayout
    private lateinit var sourceSelectionProvider: SourceSelectionProvider
    private lateinit var radioGroupDownloadStatus: RadioGroup
    private lateinit var btnReset: Button
    private lateinit var btnConfirm: Button

    private var selectedSourceProvider: SourceProvider? = null
    private var selectedDownloadStatus: DownloadStatusFilter = DownloadStatusFilter.ALL

    enum class DownloadStatusFilter {
        ALL,
        NOT_STARTED,
        DOWNLOADING,
        PAUSED,
        COMPLETED
    }

    private val sourceProviders = listOf(
        SourceProvider("HuggingFace", "hf"),
        SourceProvider("Modelscope (魔搭)", "ms"),
        SourceProvider("Modelers (魔乐)", "ml")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug_animation)

        motionLayout = findViewById(R.id.motion_layout)
        val recyclerView: RecyclerView = findViewById(R.id.recycler_view_providers)

        // Set up RecyclerView Adapter
        sourceSelectionProvider = SourceSelectionProvider(sourceProviders) { provider ->
            handleProviderSelection(provider)
        }
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = sourceSelectionProvider

        // Set default selection
        selectedSourceProvider = sourceProviders[1] // Default to Modelscope
        sourceSelectionProvider.setSelected(selectedSourceProvider!!.id)

        // Initialize filter components
        radioGroupDownloadStatus = findViewById(R.id.radio_group_download_status)
        btnReset = findViewById(R.id.btn_reset)
        btnConfirm = findViewById(R.id.btn_confirm)

        setupFilterListeners()
    }

    private fun setupFilterListeners() {
        // Handle radio button selection for download status
        radioGroupDownloadStatus.setOnCheckedChangeListener { _, checkedId ->
            selectedDownloadStatus = when (checkedId) {
                R.id.radio_download_all -> DownloadStatusFilter.ALL
                R.id.radio_download_not_started -> DownloadStatusFilter.NOT_STARTED
                R.id.radio_download_downloading -> DownloadStatusFilter.DOWNLOADING
                R.id.radio_download_paused -> DownloadStatusFilter.PAUSED
                R.id.radio_download_completed -> DownloadStatusFilter.COMPLETED
                else -> DownloadStatusFilter.ALL
            }
        }

        // Handle reset button
        btnReset.setOnClickListener {
            resetFilters()
        }

        // Handle confirm button
        btnConfirm.setOnClickListener {
            applyFilters()
        }
    }

    private fun resetFilters() {
        selectedDownloadStatus = DownloadStatusFilter.ALL
        radioGroupDownloadStatus.check(R.id.radio_download_all)
        Toast.makeText(this, R.string.filters_reset, Toast.LENGTH_SHORT).show()
    }

    private fun applyFilters() {
        // TODO: Apply filters to the model list
        // This will be implemented when connecting to ModelListFragment
        Toast.makeText(this, getString(R.string.filters_applied), Toast.LENGTH_SHORT).show()

        // Collapse the filter panel
        Handler(Looper.getMainLooper()).postDelayed({
            motionLayout.transitionToState(R.id.collapsed)
        }, 300)
    }

    private fun handleProviderSelection(sourceProvider: SourceProvider) {
        selectedSourceProvider = sourceProvider
        Toast.makeText(this, getString(R.string.starting_download_from, sourceProvider.name), Toast.LENGTH_SHORT).show()

        // Delay the collapse to show feedback
        Handler(Looper.getMainLooper()).postDelayed({
            // Transition back to the start state
            motionLayout.transitionToState(R.id.collapsed)
        }, 300)
    }
}