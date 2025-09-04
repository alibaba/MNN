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
import androidx.core.content.ContextCompat
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
        binding.startTestButtonContainer.setOnClickListener {
            Log.d(TAG, "Start test button clicked, current text: ${binding.startTestText.text}")
            presenter?.onStartBenchmarkClicked()
        }

        // Share button click handler
        binding.shareButton.setOnClickListener {
            Log.d(TAG, "Share button clicked")
            shareResultCard()
        }

        // Model selector click handler - now clicking the entire layout
        binding.modelSelectorLayout.setOnClickListener {
            // Only allow model selection in Ready and Complete states
            val currentState = getCurrentState() ?: return@setOnClickListener
            if (currentState == BenchmarkState.READY || currentState == BenchmarkState.COMPLETED) {
                showModelSelectionDialog()
            } else {
                Log.d(TAG, "Model selection disabled in state: $currentState")
                showToast(getString(R.string.cannot_change_model_during_benchmark))
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
            _binding?.modelSelectorTitle?.text = requireContext().getString(R.string.no_models_available)
            _binding?.modelSelectorStatus?.text = requireContext().getString(R.string.please_download_model)
            _binding?.modelAvatar?.setModelName("")
            _binding?.modelTagsLayout?.setTags(emptyList())
        } else {
            _binding?.modelSelectorTitle?.text = requireContext().getString(R.string.select_model_title)
            _binding?.modelSelectorStatus?.text = requireContext().getString(R.string.click_to_select_model)
            _binding?.modelAvatar?.setModelName("")
            _binding?.modelTagsLayout?.setTags(emptyList())
        }
        
        // Keep the autocomplete for compatibility
        _binding?.modelSelectorAutocomplete?.apply {
            setText(getString(R.string.select_model_title))
            isFocusable = false
            isClickable = true
        }
    }

    override fun setSelectedModel(modelWrapper: ModelItemWrapper) {
        selectedModelWrapper = modelWrapper
        
        // Update the new UI elements with model information
        val modelItem = modelWrapper.modelItem
        val modelName = modelItem.modelName ?: modelItem.modelId ?: getString(R.string.unknown_model)
        
        // Set model title and avatar
        _binding?.modelSelectorTitle?.text = modelName
        _binding?.modelAvatar?.setModelName(modelName)
        
        // Set tags similar to ModelItemHolder
        val tags = getDisplayTags(modelItem)
        _binding?.modelTagsLayout?.setTags(tags)
        
        // Set status with file size
        val formattedSize = getFormattedFileSize(modelWrapper)
        _binding?.modelSelectorStatus?.text = if (formattedSize.isNotEmpty()) {
            getString(R.string.downloaded_click_to_chat, formattedSize)
        } else {
            getString(R.string.ready_for_benchmark)
        }
        
        // Keep the autocomplete updated for compatibility
        _binding?.modelSelectorAutocomplete?.setText(modelWrapper.displayName)
        Log.d(TAG, "Selected model: ${modelWrapper.displayName}")
    }

    override fun enableStartButton(enabled: Boolean) {
        _binding?.startTestButtonContainer?.isEnabled = enabled
        _binding?.startTestButtonContainer?.alpha = if (enabled) 1.0f else 0.5f
    }

    override fun updateProgress(progress: BenchmarkProgress) {
        Log.d(TAG, "updateProgress: $progress")
        
        // Note: Do NOT update progress bar here as it's handled by UI state with realProgress
        // updateBenchmarkProgress(progress.progress) - REMOVED to avoid overriding realProgress
        
        // Update status message
        if (progress.statusMessage.isNotEmpty()) {
            updateStatusMessage(progress.statusMessage)
        }
        
        // Update test details if available
        if (progress.totalIterations > 0) {
            updateTestDetails(
                progress.currentIteration,
                progress.totalIterations,
                progress.nPrompt,
                progress.nGenerate
            )
        }
        
        // Update performance metrics if available
        if (progress.runTimeSeconds > 0) {
            updateProgressMetrics(
                progress.runTimeSeconds,
                progress.prefillTimeSeconds,
                progress.decodeTimeSeconds,
                progress.prefillSpeed,
                progress.decodeSpeed
            )
        }
    }

    override fun showResults(results: BenchmarkContract.BenchmarkResults) {
        populateResultsUI(results)
        _binding?.resultCard?.visibility = View.VISIBLE
        
        // Scroll to result_layout after showing results
        _binding?.resultLayout?.let { resultLayout ->
            resultLayout.postDelayed({
                // Get the ScrollView parent and scroll to result_layout
                val scrollView = binding.root as? android.widget.ScrollView
                scrollView?.let { sv ->
                    // Calculate the scroll position to show result_layout at the top
                    val scrollToY = resultLayout.top - sv.paddingTop
                    sv.smoothScrollTo(0, scrollToY)
                }
            }, 100) // Small delay to ensure results are fully rendered
        }
    }

    override fun hideResults() {
        _binding?.resultCard?.visibility = View.GONE
    }

    override fun updateStatus(message: String) {
        _binding?.statusMessage?.text = message
        _binding?.statusCard?.visibility = View.VISIBLE
    }

    override fun hideStatus() {
        _binding?.statusCard?.visibility = View.GONE
    }

    override fun setStartButtonText(text: String) {
        Log.d(TAG, "Setting start button text to: $text")
        if (isFragmentValid()) {
            binding.startTestText.text = text
            
            // Update button icon and background based on text
            when (text) {
                getString(R.string.start_test) -> {
                    binding.startTestIcon.setImageResource(R.drawable.ic_play_fill)
                    binding.startTestIcon.visibility = View.VISIBLE
                    binding.startTestProgress.visibility = View.GONE
                    binding.startTestArrow.visibility = View.VISIBLE
                    binding.startTestButtonContainer.background = ContextCompat.getDrawable(requireContext(), R.drawable.benchmark_button_background_selector)
                }
                getString(R.string.restart_test) -> {
                    binding.startTestIcon.setImageResource(R.drawable.ic_play_fill)
                    binding.startTestIcon.visibility = View.VISIBLE
                    binding.startTestProgress.visibility = View.GONE
                    binding.startTestArrow.visibility = View.VISIBLE
                    binding.startTestButtonContainer.background = ContextCompat.getDrawable(requireContext(), R.drawable.benchmark_button_background_selector)
                }
                getString(R.string.stop_test) -> {
                    // Check if we're in a running state (progress card is visible)
                    if (binding.progressCard.visibility == View.VISIBLE) {
                        // Show progress indicator when actively running (like iOS)
                        binding.startTestIcon.visibility = View.GONE
                        binding.startTestProgress.visibility = View.VISIBLE
                        binding.startTestArrow.visibility = View.GONE
                    } else {
                        // Show stop icon when just stopping/initializing
                        binding.startTestIcon.setImageResource(R.drawable.ic_stop_fill)
                        binding.startTestIcon.visibility = View.VISIBLE
                        binding.startTestProgress.visibility = View.GONE
                        binding.startTestArrow.visibility = View.GONE
                    }
                    binding.startTestButtonContainer.background = ContextCompat.getDrawable(requireContext(), R.drawable.benchmark_button_stop_background_selector)
                }
                else -> {
                    // For other text (like "Share", "Upload to Leaderboard"), hide icon and arrow
                    binding.startTestIcon.visibility = View.GONE
                    binding.startTestProgress.visibility = View.GONE
                    binding.startTestArrow.visibility = View.VISIBLE
                    binding.startTestButtonContainer.background = ContextCompat.getDrawable(requireContext(), R.drawable.benchmark_button_background_selector)
                }
            }
        }
    }

    override fun setStartButtonEnabled(enabled: Boolean) {
        if (isFragmentValid()) {
            binding.startTestButtonContainer.isEnabled = enabled
            binding.startTestButtonContainer.alpha = if (enabled) 1.0f else 0.5f
        }
    }

    override fun showProgressBar() {
//        binding.progressBar.visibility = View.VISIBLE
    }

    override fun hideProgressBar() {
//        binding.progressBar.visibility = View.GONE
        // Hide status card if results are visible
        if (_binding?.resultCard?.visibility == View.VISIBLE) {
            _binding?.statusCard?.visibility = View.INVISIBLE
        }
    }

    override fun showBenchmarkIcon(show: Boolean) {
        // Benchmark icon removed to match iOS - no large icon display
        Log.d(TAG, "showBenchmarkIcon: $show (removed to match iOS)")
    }

    override fun showBenchmarkProgressBar(show: Boolean) {
        // Progress bar removed with benchmark icon - using progress card instead
        // Update button state for consistency
        if (isFragmentValid()) {
            updateButtonIconState()
        }
        Log.d(TAG, "showBenchmarkProgressBar: $show (using progress card instead)")
    }

    override fun updateBenchmarkProgress(progress: Int) {
        if (isFragmentValid()) {
            // Update progress percentage text
            binding.progressPercentage.text = "$progress%"
            // Update actual progress bar
            binding.progressBar.progress = progress
        }
        Log.d(TAG, "updateBenchmarkProgress: $progress% (updated progress bar)")
    }
    
    private fun updateButtonIconState() {
        if (isFragmentValid()) {
            val currentText = binding.startTestText.text.toString()
            if (currentText == getString(R.string.stop_test)) {
                // Re-evaluate the stop button state based on progress card visibility
                if (binding.progressCard.visibility == View.VISIBLE) {
                    // Show progress indicator when actively running (like iOS)
                    binding.startTestIcon.visibility = View.GONE
                    binding.startTestProgress.visibility = View.VISIBLE
                    binding.startTestArrow.visibility = View.GONE
                } else {
                    // Show stop icon when just stopping/initializing
                    binding.startTestIcon.setImageResource(R.drawable.ic_stop_fill)
                    binding.startTestIcon.visibility = View.VISIBLE
                    binding.startTestProgress.visibility = View.GONE
                    binding.startTestArrow.visibility = View.GONE
                }
            }
        }
    }

    override fun showProgressCard(show: Boolean) {
        if (isFragmentValid()) {
            binding.progressCard.visibility = if (show) View.VISIBLE else View.GONE
        }
        Log.d(TAG, "showProgressCard: $show")
    }

    override fun showStatusCard(show: Boolean) {
        if (isFragmentValid()) {
            binding.statusCard.visibility = if (show) View.VISIBLE else View.GONE
        }
        Log.d(TAG, "showStatusCard: $show")
    }

    override fun updateStatusMessage(message: String) {
        if (isFragmentValid()) {
            binding.statusMessage.text = message
        }
        Log.d(TAG, "updateStatusMessage: $message")
    }

    override fun updateTestDetails(
        currentIteration: Int,
        totalIterations: Int,
        nPrompt: Int,
        nGenerate: Int
    ) {
        if (isFragmentValid()) {
            binding.testIterationInfo.text = "Test $currentIteration of $totalIterations"
            binding.testConfigInfo.text = "PP: $nPrompt • TG: $nGenerate"
            binding.testDetailsContainer.visibility = View.VISIBLE
        }
        Log.d(TAG, "updateTestDetails: $currentIteration/$totalIterations, PP: $nPrompt, TG: $nGenerate")
    }

    override fun updateProgressMetrics(
        runtime: Float,
        prefillTime: Float,
        decodeTime: Float,
        prefillSpeed: Float,
        decodeSpeed: Float
    ) {
        if (isFragmentValid()) {
            binding.runtimeMetric.updateMetric("Runtime", String.format("%.3fs", runtime), "ic_clock")
            binding.prefillTimeMetric.updateMetric("Prefill", String.format("%.3fs", prefillTime), "ic_arrow_up_circle")
            binding.decodeTimeMetric.updateMetric("Decode", String.format("%.3fs", decodeTime), "ic_arrow_down_circle")
            binding.prefillSpeedMetricProgress.updateMetric("Prefill Speed", String.format("%.2f t/s", prefillSpeed), "ic_speedometer")
            binding.decodeSpeedMetricProgress.updateMetric("Decode Speed", String.format("%.2f t/s", decodeSpeed), "ic_gauge")
        }
        Log.d(TAG, "updateProgressMetrics: runtime=$runtime, prefill=$prefillTime, decode=$decodeTime")
    }

    override fun enableModelSelector(enabled: Boolean) {
        if (isFragmentValid()) {
            binding.modelSelectorLayout.isEnabled = enabled
            binding.modelSelectorLayout.alpha = if (enabled) 1.0f else 0.6f
        }
        Log.d(TAG, "enableModelSelector: $enabled")
    }

    override fun showBackButton(show: Boolean) {
        // Back button removed, no longer needed
        Log.d(TAG, "showBackButton: $show (back button removed)")
    }

    override fun showModelSelectorCard(show: Boolean) {
        if (isFragmentValid()) {
            binding.modelSelectorCard.visibility = if (show) View.VISIBLE else View.GONE
        }
        Log.d(TAG, "showModelSelectorCard: $show")
    }

    override fun updateButtonLayout(showBackButton: Boolean) {
        // Back button removed, main button always full width
        Log.d(TAG, "updateButtonLayout: showBackButton=$showBackButton (back button removed)")
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
            showToast(getString(R.string.failed_to_share_result, e.message ?: getString(R.string.unknown_error)))
        }
    }

    override fun uploadToLeaderboard() {
        presenter?.onUploadToLeaderboardClicked()
    }
    
    override fun showUploadProgress(message: String) {
        _binding?.statusMessage?.text = message
        _binding?.statusCard?.visibility = View.VISIBLE
        // Disable the upload button while uploading
        _binding?.startTestButtonContainer?.isEnabled = false
        _binding?.startTestButtonContainer?.alpha = 0.5f
        Log.d(TAG, "Showing upload progress: $message")
    }
    
    override fun hideUploadProgress() {
        _binding?.statusCard?.visibility = View.GONE
        // Re-enable the upload button
        _binding?.startTestButtonContainer?.isEnabled = true
        _binding?.startTestButtonContainer?.alpha = 1.0f
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
        _binding?.resultCard?.visibility = View.VISIBLE
        _binding?.modelName?.text = results.modelDisplayName
        DeviceName.with(requireContext()).request { info, error ->
            val deviceName = info?.marketName ?: info?.name ?: android.os.Build.MODEL
            _binding?.deviceInfo?.text = getString(R.string.benchmark_device_info, deviceName)
        }
        
        // Use BenchmarkResultsHelper to process test results
        val statistics = BenchmarkResultsHelper.processTestResults(requireContext(), results.testResults)
        
        // Display configuration
        _binding?.benchmarkConfigText?.text = statistics.configText
        
        // Set up performance metrics using new PerformanceMetricView components
        _binding?.prefillSpeedMetric?.setSpeedMetric(
            R.drawable.ic_speed,
            R.string.prefill_speed_title,
            statistics.prefillStats,
            R.color.benchmark_gradient_start
        )
        
        _binding?.decodeSpeedMetric?.setSpeedMetric(
            R.drawable.ic_gauge,
            R.string.decode_speed_title, 
            statistics.decodeStats,
            R.color.benchmark_gradient_end
        )
        
        // Set up total time metric
        _binding?.totalTokensMetric?.setTotalTimeMetric(
            statistics.totalTimeSeconds,
            R.color.benchmark_success
        )
        
        // Set up peak memory metric
        val totalMemoryKb = BenchmarkResultsHelper.getTotalMemoryKb()
        _binding?.peakMemoryMetric?.setMemoryMetric(
            results.maxMemoryKb,
            totalMemoryKb,
            R.color.benchmark_warning
        )
        
        // Timestamp
        _binding?.timestamp?.text = results.timestamp
        
        Log.d(TAG, "Results populated - Memory: ${results.maxMemoryKb} KB, Model: ${results.modelDisplayName}")
    }

    // ===== Helper Methods =====

    /**
     * Check if fragment is in valid state for UI updates
     */
    private fun isFragmentValid(): Boolean {
        return isAdded && !isDetached && _binding != null
    }

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