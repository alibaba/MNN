package com.alibaba.mnnllm.android.benchmark

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import java.io.FileOutputStream
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentBenchmarkBinding
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.chat.SelectModelFragment
import com.jaredrummler.android.device.DeviceName
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.utils.FileUtils
import java.io.File
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class BenchmarkFragment : Fragment(), BenchmarkContract.View {

    companion object {
        private const val TAG = "BenchmarkFragment"
    }

    private var _binding: FragmentBenchmarkBinding? = null
    private val binding get() = _binding!!
    
    // MVP Components
    private var presenter: BenchmarkPresenter? = null
    private var selectedModelWrapper: ModelItemWrapper? = null
    private var availableModels: List<ModelItemWrapper> = emptyList()

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

        // Back button click handler
        binding.backButton.setOnClickListener {
            Log.d(TAG, "Back button clicked")
            presenter?.onBackClicked()
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
        val modelFilter: (ModelItemWrapper) -> Boolean = { modelWrapper ->
            !ModelUtils.isDiffusionModel(modelWrapper.displayName)
        }
        val selectModelFragment = SelectModelFragment.newInstance(availableModels, modelFilter, currentModelId)
        selectModelFragment.setOnModelSelectedListener { modelWrapper ->
            presenter?.onModelSelected(modelWrapper)
        }
        selectModelFragment.show(parentFragmentManager, SelectModelFragment.TAG)
    }

    // ===== BenchmarkContract.View Implementation =====

    override fun showLoading() {
//        binding.progressBar.visibility = View.VISIBLE
    }

    override fun hideLoading() {
//        binding.progressBar.visibility = View.GONE
    }

    override fun showError(message: String) {
        showToast(message)
    }

    override fun showToast(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    override fun showStopConfirmationDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.benchmark_stop_confirmation_title))
            .setMessage(getString(R.string.benchmark_stop_confirmation_message))
            .setPositiveButton(getString(R.string.yes)) { _, _ ->
                presenter?.onStopBenchmarkClicked()
            }
            .setNegativeButton(getString(R.string.no), null)
            .show()
    }

    override fun updateModelSelector(models: List<ModelItemWrapper>) {
        availableModels = models
        
        // Update the new UI elements
        if (models.isEmpty()) {
            binding.modelSelectorTitle.text = requireContext().getString(R.string.no_models_available)
            binding.modelSelectorStatus.text = requireContext().getString(R.string.please_download_model)
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

    override fun setSelectedModel(modelWrapper: ModelItemWrapper) {
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
        binding.resultCard.visibility = View.INVISIBLE
    }

    override fun showResults(results: BenchmarkContract.BenchmarkResults) {
        populateResultsUI(results)
        binding.resultCard.visibility = View.VISIBLE
    }

    override fun hideResults() {
        binding.testResultsTitle.visibility = View.INVISIBLE
        binding.resultCard.visibility = View.INVISIBLE
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
//        binding.progressBar.visibility = View.GONE
        // Hide textStatus if results are visible
        if (binding.resultCard.visibility == View.VISIBLE) {
            binding.textStatus.visibility = View.INVISIBLE
        }
    }

    override fun showBenchmarkIcon(show: Boolean) {
        binding.iconBenchmark.visibility = if (show) View.VISIBLE else View.INVISIBLE
        binding.iconBenchmarkParent.visibility = if (show) View.VISIBLE else View.INVISIBLE
        Log.d(TAG, "showBenchmarkIcon: $show")
    }

    override fun showBenchmarkProgressBar(show: Boolean) {
        binding.benchmarkProgressBar.visibility = if (show) View.VISIBLE else View.INVISIBLE
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

    override fun showBackButton(show: Boolean) {
        binding.backButton.visibility = if (show) View.VISIBLE else View.GONE
        Log.d(TAG, "showBackButton: $show")
    }

    override fun showModelSelectorCard(show: Boolean) {
        binding.modelSelectorCard.visibility = if (show) View.VISIBLE else View.GONE
        Log.d(TAG, "showModelSelectorCard: $show")
    }

    override fun updateButtonLayout(showBackButton: Boolean) {
        if (showBackButton) {
            // Show back button and adjust main button layout
            binding.backButton.visibility = View.VISIBLE
            binding.startTestButton.layoutParams = LinearLayout.LayoutParams(
                0, 
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                weight = 1f
                marginStart = 8
            }
        } else {
            // Hide back button and make main button full width
            binding.backButton.visibility = View.GONE
            binding.startTestButton.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        Log.d(TAG, "updateButtonLayout: showBackButton=$showBackButton")
    }

    override fun shareResultCard() {
        try {
            // Create bitmap from result card
            val bitmap = createBitmapFromView(binding.resultCard)
            
            // Save bitmap to cache directory
            val cachePath = java.io.File(requireContext().cacheDir, "shared_images")
            cachePath.mkdirs()
            val file = java.io.File(cachePath, "benchmark_result.png")
            
            val stream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            stream.close()
            
            // Create content URI using FileProvider
            val contentUri = FileProvider.getUriForFile(
                requireContext(),
                "${requireContext().packageName}.fileprovider",
                file
            )
            
            // Create share intent
            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                putExtra(Intent.EXTRA_TEXT, getString(R.string.share_benchmark_result))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            
            startActivity(Intent.createChooser(shareIntent, getString(R.string.share)))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error sharing result card", e)
            showToast("Failed to share result: ${e.message}")
        }
    }

    override fun uploadToLeaderboard() {
        presenter?.onUploadToLeaderboardClicked()
    }
    
    override fun showUploadProgress(message: String) {
        binding.textStatus.text = message
        binding.textStatus.visibility = View.VISIBLE
        // Disable the upload button while uploading
        binding.startTestButton.isEnabled = false
        Log.d(TAG, "Showing upload progress: $message")
    }
    
    override fun hideUploadProgress() {
        binding.textStatus.visibility = View.GONE
        // Re-enable the upload button
        binding.startTestButton.isEnabled = true
        Log.d(TAG, "Hiding upload progress")
    }
    
    override fun showRankInfo(rank: Int, totalUsers: Int) {
        val rankMessage = if (rank > 0) {
            getString(R.string.leaderboard_rank_info, rank, totalUsers)
        } else {
            getString(R.string.leaderboard_rank_not_found)
        }
        
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.leaderboard_ranking))
            .setMessage(rankMessage)
            .setPositiveButton(getString(R.string.ok), null)
            .show()
        
        Log.d(TAG, "Showing rank info: rank=$rank, totalUsers=$totalUsers")
    }

    private fun createBitmapFromView(view: View): Bitmap {
        val bitmap = Bitmap.createBitmap(
            view.width, 
            view.height, 
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bitmap)
        view.draw(canvas)
        return bitmap
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
//        binding.benchmarkConfig.text = statistics.configText
        
        // Show prompt processing results (prefill)
        statistics.prefillStats?.let { stats ->
            // Display average value
            val averageText = "%.2f".format(stats.average)
            // Display standard deviation in label
            val labelText = "tokens/s ±%.2f".format(stats.stdev)
            Log.d(TAG, "Setting prefill - Average: '$averageText', Label: '$labelText'")
            binding.promptProcessingValue.text = averageText
            binding.promptProcessingLabel.text = labelText
        } ?: run {
            Log.d(TAG, "No prefill stats available")
        }
        
        // Show token generation results (decode)
        statistics.decodeStats?.let { stats ->
            // Display average value
            val averageText = "%.2f".format(stats.average)
            // Display standard deviation in label
            val labelText = "tokens/s ±%.2f".format(stats.stdev)
            Log.d(TAG, "Setting decode - Average: '$averageText', Label: '$labelText'")
            binding.tokenGenerationValue.text = averageText
            binding.tokenGenerationLabel.text = labelText
        } ?: run {
            Log.d(TAG, "No decode stats available")
        }
        
        // Display peak memory usage
        val (peakValue, maxValue) = BenchmarkResultsHelper.formatMemoryUsageDetailed(requireContext(), results.maxMemoryKb)
        binding.peakMemoryValue.text = peakValue
        binding.maxMemoryValue.text = maxValue
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
        return modelItem.getDisplayTags(requireContext()).take(3)
    }

    /**
     * Get formatted file size for model, similar to ModelItemHolder
     */
    private fun getFormattedFileSize(modelWrapper: ModelItemWrapper): String {
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

    override fun onHiddenChanged(hidden: Boolean) {
        super.onHiddenChanged(hidden)
        if (!hidden) {
            presenter?.loadModels()
        }
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