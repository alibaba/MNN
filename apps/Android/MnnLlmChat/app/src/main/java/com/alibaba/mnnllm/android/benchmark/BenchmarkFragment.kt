package com.alibaba.mnnllm.android.benchmark

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentBenchmarkBinding
import com.alibaba.mnnllm.android.utils.ModelListManager
import com.alibaba.mnnllm.android.chat.SelectModelFragment
import com.jaredrummler.android.device.DeviceName
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.utils.FileUtils
import java.io.File

class BenchmarkFragment : Fragment(), BenchmarkContract.View {

    companion object {
        private const val TAG = "BenchmarkFragment"
    }

    private var _binding: FragmentBenchmarkBinding? = null
    private val binding get() = _binding!!
    
    // MVP Components
    private var presenter: BenchmarkPresenter? = null
    private var selectedModelWrapper: ModelListManager.ModelItemWrapper? = null
    private var availableModels: List<ModelListManager.ModelItemWrapper> = emptyList()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentBenchmarkBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Initialize presenter
        presenter = BenchmarkPresenter(requireContext(), this, lifecycleScope)
        
        setupClickListeners()
        presenter?.onCreate()
    }

    private fun setupClickListeners() {
        binding.startTestButton.setOnClickListener {
            Log.d(TAG, "Start test button clicked, current text: ${binding.startTestButton.text}")
            presenter?.onStartBenchmarkClicked()
        }

        binding.submitButton.setOnClickListener {
            presenter?.onSubmitResultClicked()
        }

        // Model selector click handler - now clicking the entire layout
        binding.modelSelectorLayout.setOnClickListener {
            // Only allow model selection in Ready and Complete states
            val currentState = getCurrentState() ?: return@setOnClickListener
            if (currentState == BenchmarkState.READY || currentState == BenchmarkState.COMPLETED) {
                showModelSelectionDialog()
            } else {
                Log.d(TAG, "Model selection disabled in state: $currentState")
                showToast("Cannot change model during benchmark")
            }
        }
        
        // Keep the autocomplete click handler for compatibility
        binding.modelSelectorAutocomplete.setOnClickListener {
            val currentState = getCurrentState() ?: return@setOnClickListener
            if (currentState == BenchmarkState.READY || currentState == BenchmarkState.COMPLETED) {
                showModelSelectionDialog()
            }
        }
    }

    private fun showModelSelectionDialog() {
        val currentModelId = selectedModelWrapper?.modelItem?.modelId
        val selectModelFragment = SelectModelFragment.newInstance(availableModels, null, currentModelId)
        selectModelFragment.setOnModelSelectedListener { modelWrapper ->
            presenter?.onModelSelected(modelWrapper)
        }
        selectModelFragment.show(parentFragmentManager, SelectModelFragment.TAG)
    }

    // ===== BenchmarkContract.View Implementation =====

    override fun showLoading() {
        binding.progressBar.visibility = View.VISIBLE
    }

    override fun hideLoading() {
        binding.progressBar.visibility = View.GONE
    }

    override fun showError(message: String) {
        showToast(message)
    }

    override fun showToast(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    override fun showStopConfirmationDialog() {
        AlertDialog.Builder(requireContext())
            .setTitle(getString(R.string.benchmark_stop_confirmation_title))
            .setMessage(getString(R.string.benchmark_stop_confirmation_message))
            .setPositiveButton(getString(R.string.yes)) { _, _ ->
                presenter?.onStopBenchmarkClicked()
            }
            .setNegativeButton(getString(R.string.no), null)
            .show()
    }

    override fun updateModelSelector(models: List<ModelListManager.ModelItemWrapper>) {
        availableModels = models
        
        // Update the new UI elements
        if (models.isEmpty()) {
            binding.modelSelectorTitle.text = "No models available"
            binding.modelSelectorStatus.text = "Please download a model first"
            binding.modelAvatar.setModelName("")
            binding.modelTagsLayout.setTags(emptyList())
        } else {
            binding.modelSelectorTitle.text = "Select Model"
            binding.modelSelectorStatus.text = "Click to select model"
            binding.modelAvatar.setModelName("")
            binding.modelTagsLayout.setTags(emptyList())
        }
        
        // Keep the autocomplete for compatibility
        binding.modelSelectorAutocomplete.apply {
            setText("Select Model")
            isFocusable = false
            isClickable = true
        }
    }

    override fun setSelectedModel(modelWrapper: ModelListManager.ModelItemWrapper) {
        selectedModelWrapper = modelWrapper
        
        // Update the new UI elements with model information
        val modelItem = modelWrapper.modelItem
        val modelName = modelItem.modelName ?: modelItem.modelId ?: "Unknown Model"
        
        // Set model title and avatar
        binding.modelSelectorTitle.text = modelName
        binding.modelAvatar.setModelName(modelName)
        
        // Set tags similar to ModelItemHolder
        val tags = getDisplayTags(modelItem)
        binding.modelTagsLayout.setTags(tags)
        
        // Set status with file size
        val formattedSize = getFormattedFileSize(modelWrapper)
        binding.modelSelectorStatus.text = if (formattedSize.isNotEmpty()) {
            getString(R.string.downloaded_click_to_chat, formattedSize)
        } else {
            "Ready for benchmark"
        }
        
        // Keep the autocomplete updated for compatibility
        binding.modelSelectorAutocomplete.setText(modelWrapper.displayName)
        Log.d(TAG, "Selected model: ${modelWrapper.displayName}")
    }

    override fun enableStartButton(enabled: Boolean) {
        binding.startTestButton.isEnabled = enabled
    }

    override fun updateProgress(progress: BenchmarkProgress) {
        binding.textStatus.text = progress.statusMessage
        binding.textStatus.visibility = View.VISIBLE
        binding.resultCard.visibility = View.GONE
    }

    override fun showResults(results: BenchmarkContract.BenchmarkResults) {
        populateResultsUI(results)
        binding.resultCard.visibility = View.VISIBLE
    }

    override fun hideResults() {
        binding.testResultsTitle.visibility = View.GONE
        binding.resultCard.visibility = View.GONE
    }

    override fun updateStatus(message: String) {
        binding.textStatus.text = message
        binding.textStatus.visibility = View.VISIBLE
    }

    override fun hideStatus() {
        binding.textStatus.visibility = View.GONE
    }

    override fun setStartButtonText(text: String) {
        Log.d(TAG, "Setting start button text to: $text")
        binding.startTestButton.text = text
    }

    override fun setStartButtonEnabled(enabled: Boolean) {
        binding.startTestButton.isEnabled = enabled
    }

    override fun showProgressBar() {
//        binding.progressBar.visibility = View.VISIBLE
    }

    override fun hideProgressBar() {
        binding.progressBar.visibility = View.GONE
        // Hide textStatus if results are visible
        if (binding.resultCard.visibility == View.VISIBLE) {
            binding.textStatus.visibility = View.GONE
        }
    }

    override fun showBenchmarkIcon(show: Boolean) {
        binding.iconBenchmark.visibility = if (show) View.VISIBLE else View.GONE
        binding.iconBenchmarkParent.visibility = if (show) View.VISIBLE else View.GONE
        Log.d(TAG, "showBenchmarkIcon: $show")
    }

    override fun showBenchmarkProgressBar(show: Boolean) {
        binding.benchmarkProgressBar.visibility = if (show) View.VISIBLE else View.GONE
        Log.d(TAG, "showBenchmarkProgressBar: $show")
    }

    override fun updateBenchmarkProgress(progress: Int) {
        binding.benchmarkProgressBar.progress = progress
        Log.d(TAG, "updateBenchmarkProgress: $progress%")
    }

    override fun enableModelSelector(enabled: Boolean) {
        binding.modelSelectorLayout.isEnabled = enabled
        binding.modelSelectorLayout.alpha = if (enabled) 1.0f else 0.6f
        Log.d(TAG, "enableModelSelector: $enabled")
    }

    // ===== UI Helpers =====

    private fun populateResultsUI(results: BenchmarkContract.BenchmarkResults) {
        binding.resultCard.visibility = View.VISIBLE
        binding.testResultsTitle.visibility = View.VISIBLE
        binding.modelName.text = results.modelDisplayName
        DeviceName.with(requireContext()).request { info, error ->
            val deviceName = info?.marketName ?: info?.name ?: android.os.Build.MODEL
            binding.deviceInfo.text = getString(R.string.benchmark_device_info, deviceName)
        }
        
        // Use BenchmarkResultsHelper to process test results
        val statistics = BenchmarkResultsHelper.processTestResults(requireContext(), results.testResults)
        
        // Display configuration
        binding.benchmarkConfig.text = statistics.configText
        
        // Show prompt processing results (prefill)
        statistics.prefillStats?.let { stats ->
            val valueText = BenchmarkResultsHelper.formatSpeedStatisticsLine(stats)
            val labelText = BenchmarkResultsHelper.formatSpeedLabelOnly(stats)
            Log.d(TAG, "Setting prefill - Value: '$valueText', Label: '$labelText'")
            binding.promptProcessingValue.text = valueText
            binding.promptProcessingLabel.text = ""
        } ?: run {
            Log.d(TAG, "No prefill stats available")
        }
        
        // Show token generation results (decode)
        statistics.decodeStats?.let { stats ->
            val valueText = BenchmarkResultsHelper.formatSpeedStatisticsLine(stats)
            val labelText = BenchmarkResultsHelper.formatSpeedLabelOnly(stats)
            Log.d(TAG, "Setting decode - Value: '$valueText', Label: '$labelText'")
            binding.tokenGenerationValue.text = valueText
            binding.tokenGenerationLabel.text = ""
        } ?: run {
            Log.d(TAG, "No decode stats available")
        }
        
        // Model parameters info
        binding.modelParams.text = BenchmarkResultsHelper.formatModelParams(
            requireContext(),
            statistics.totalTokensProcessed, 
            statistics.totalTests
        )
        
        // Display peak memory usage
        val (memValue, memLabel) = BenchmarkResultsHelper.formatMemoryUsageDetailed(requireContext(), results.maxMemoryKb)
        binding.peakMemoryLabel.text = memLabel
        // Timestamp
        binding.timestamp.text = results.timestamp
        
        Log.d(TAG, "Results populated - Memory: ${results.maxMemoryKb} KB, Model: ${results.modelDisplayName}")
    }

    // ===== Helper Methods =====

    /**
     * Get current benchmark state from presenter
     */
    private fun getCurrentState(): BenchmarkState? {
        return presenter?.getCurrentState()
    }

    /**
     * Get display tags for model, similar to ModelItemHolder
     */
    private fun getDisplayTags(modelItem: ModelItem): List<String> {
        val tags = mutableListOf<String>()
        
        // Add source tag first
        val source = getModelSource(modelItem.modelId)
        if (source != null) {
            tags.add(source)
        }
        
        // Use getTags() which now prioritizes market tags from model_market.json
        val marketTags = modelItem.getTags()
        
        // Add local/downloaded status
        if (modelItem.isLocal) {
            tags.add(getString(R.string.local))
        } else if (marketTags.isNotEmpty()) {
            // If we have market tags, use them directly (they're already user-friendly)
            tags.addAll(marketTags.take(2)) // Limit to 2 market tags to leave room for source tag
        }
        
        // Limit total tags to 3 for better UI layout
        return tags.take(3)
    }

    /**
     * Extract source information from modelId
     */
    private fun getModelSource(modelId: String?): String? {
        return when {
            modelId == null -> null
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") -> getString(R.string.huggingface)
            modelId.startsWith("ModelScope/") -> getString(R.string.modelscope)
            modelId.startsWith("Modelers/") -> getString(R.string.modelers)
            else -> null
        }
    }

    /**
     * Get formatted file size for model, similar to ModelItemHolder
     */
    private fun getFormattedFileSize(modelWrapper: ModelListManager.ModelItemWrapper): String {
        val modelItem = modelWrapper.modelItem
        val modelDownloadManager = ModelDownloadManager.getInstance(requireContext())
        
        // Try to get file size using the same method as MarketItemHolder
        modelItem.modelId?.let { modelId ->
            val downloadedFile = modelDownloadManager.getDownloadedFile(modelId)
            if (downloadedFile != null) {
                return FileUtils.getFileSizeString(downloadedFile)
            }
        }
        
        // Fallback to direct file size calculation
        if (modelWrapper.downloadSize > 0) {
            return FileUtils.formatFileSize(modelWrapper.downloadSize)
        }
        
        // Try to get size from local path
        modelItem.localPath?.let { localPath ->
            val file = File(localPath)
            if (file.exists()) {
                return FileUtils.getFileSizeString(file)
            }
        }
        
        return ""
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
    
    override fun onDestroy() {
        super.onDestroy()
        presenter?.onDestroy()
    }
}