package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.Log
import androidx.lifecycle.LifecycleCoroutineScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ModelListManager
import kotlinx.coroutines.launch

/**
 * Presenter for benchmark functionality
 * Coordinates between BenchmarkView and BenchmarkModel
 */
class BenchmarkPresenter(
    private val context: Context,
    private val view: BenchmarkContract.View,
    private val lifecycleScope: LifecycleCoroutineScope
) : BenchmarkContract.Presenter {
    
    companion object {
        private const val TAG = "BenchmarkPresenter"
    }
    
    private val model = BenchmarkModel()
    private var selectedModelWrapper: ModelListManager.ModelItemWrapper? = null
    private var availableModels: List<ModelListManager.ModelItemWrapper> = emptyList()
    private var isStopping = false // Flag to prevent UI updates after stop
    
    override fun onCreate() {
        setupModelSelector()
    }
    
    override fun onDestroy() {
        model.release()
    }
    
    override fun onStartBenchmarkClicked() {
        Log.d(TAG, "onStartBenchmarkClicked called, isBenchmarkRunning: ${model.isBenchmarkRunning()}")
        if (!model.isBenchmarkRunning()) {
            startBenchmark()
        } else {
            view.showStopConfirmationDialog()
        }
    }
    
    override fun onStopBenchmarkClicked() {
        stopBenchmark()
    }
    
    override fun onModelSelected(modelWrapper: ModelListManager.ModelItemWrapper) {
        selectedModelWrapper = modelWrapper
        view.setSelectedModel(modelWrapper)
        Log.d(TAG, "Selected model: ${modelWrapper.displayName}, isLocal: ${modelWrapper.modelItem.isLocal}, localPath: ${modelWrapper.modelItem.localPath}")
    }
    
    override fun onDeleteResultClicked() {
        view.hideResults()
        view.hideStatus()
        view.showToast("Result deleted")
    }
    
    override fun onSubmitResultClicked() {
    }
    
    override fun onViewLeaderboardClicked() {
    }
    
    /**
     * Setup model selector with available models
     */
    private fun setupModelSelector() {
        lifecycleScope.launch {
            try {
                availableModels = model.loadAvailableModels(context)
                view.updateModelSelector(availableModels)
                
                // Enable the start button only when models are loaded
                view.enableStartButton(availableModels.isNotEmpty())
                
                if (availableModels.isEmpty()) {
                    view.showError("No models available. Please download a model first.")
                } else {
                    // Set first model as default
                    selectedModelWrapper = availableModels.firstOrNull()
                    selectedModelWrapper?.let { wrapper ->
                        view.setSelectedModel(wrapper)
                    }
                }
            } catch (e: Exception) {
                view.showError("Failed to load models: ${e.message}")
                view.enableStartButton(false)
            }
        }
    }
    
    /**
     * Start benchmark test
     */
    private fun startBenchmark() {
        val modelWrapper = selectedModelWrapper
        if (modelWrapper == null) {
            view.showError("Please select a model")
            return
        }
        
        // Initialize UI for benchmark test
        onBenchmarkStarted()
        
        lifecycleScope.launch {
            try {
                // Initialize model if not already done
                if (!model.isModelInitialized() || 
                    model.getModelInfo() != modelWrapper.modelItem.modelId) {
                    
                    Log.d(TAG, "Model needs initialization")
                    view.updateStatus(context.getString(R.string.benchmark_loading_model))
                    
                    val configPath = if (modelWrapper.modelItem.isLocal && !modelWrapper.modelItem.localPath.isNullOrEmpty()) {
                        "${modelWrapper.modelItem.localPath}/config.json"
                    } else {
                        null
                    }
                    
                    Log.d(TAG, "Initializing model with config: $configPath")
                    val initSuccess = model.initializeModel(modelWrapper.modelItem.modelId!!, configPath)
                    
                    if (!initSuccess) {
                        Log.e(TAG, "Model initialization failed - resetting UI state")
                        view.showError("Failed to initialize model")
                        resetUIState(force = true)
                        return@launch
                    }
                    Log.d(TAG, "Model initialization successful")
                } else {
                    Log.d(TAG, "Model already initialized, skipping init")
                }
                
                // Start benchmark
                model.startBenchmark(
                    context,
                    modelWrapper,
                    object : BenchmarkModel.BenchmarkModelCallback {
                        override fun onProgress(progress: BenchmarkProgress) {
                            // Check if benchmark was stopped, if so, ignore progress updates
                            if (isStopping) {
                                Log.d(TAG, "Ignoring progress update as benchmark is stopping: ${progress.statusMessage}")
                                return
                            }
                            
                            // Format progress message based on structured data
                            val formattedProgress = formatProgressMessage(progress)
                            view.updateProgress(formattedProgress)
                            Log.d(TAG, "Benchmark Progress (${progress.progress}%): ${formattedProgress.statusMessage}")
                        }
                        
                        override fun onComplete(results: BenchmarkContract.BenchmarkResults) {
                            Log.d(TAG, "Benchmark onComplete received: $results")
                            // Accumulate results and wait for actual completion below
                            view.updateStatus(context.getString(R.string.benchmark_processing_results))
                            view.showResults(results)
                            view.setStartButtonText(context.getString(R.string.start_test))
                            // Do not reset UI here to avoid premature reset
                            view.hideStatus()
                        }
                        
                        override fun onBenchmarkModelError(message: String) {
                            view.updateStatus(context.getString(R.string.benchmark_failed, message))
                            view.showError("Benchmark failed: $message")
                            resetUIState(force = true)
                        }
                    }
                )
//
//                // Launch a watcher coroutine that waits until the benchmark actually finishes
//                launch {
//                    while (model.isBenchmarkRunning()) {
//                        kotlinx.coroutines.delay(1000)
//                    }
//                    // Once running flag is false, finalize UI
//                    resetUIState(force = true)
//                }
            } catch (e: Exception) {
                view.updateStatus(context.getString(R.string.benchmark_failed, e.message ?: "Unknown error"))
                view.showError("Error: ${e.message}")
                resetUIState(force = true)
            }
        }
    }
    
    /**
     * Stop benchmark test
     */
    private fun stopBenchmark() {
        // Set stopping flag to prevent further UI updates
        isStopping = true
        
        // Show stopping status
        view.updateStatus(context.getString(R.string.benchmark_stopping))
        
        // Stop the benchmark. The watcher coroutine in startBenchmark
        // will handle resetting the UI once model.isBenchmarkRunning() returns false.
        model.stopBenchmark()
    }
    
    /**
     * Initialize UI when benchmark starts
     */
    private fun onBenchmarkStarted() {
        Log.d(TAG, "onBenchmarkStarted called")
        
        // Reset stopping flag when starting a new benchmark
        isStopping = false
        
        val stopText = context.getString(R.string.stop_test)
        Log.d(TAG, "Setting button text to: $stopText")
        view.setStartButtonText(stopText)
        view.setStartButtonEnabled(true)
        view.showProgressBar()
        view.hideResults()
        view.updateStatus(context.getString(R.string.benchmark_initializing))
    }
    
    /**
     * Reset UI state after benchmark completes or stops
     */
    private fun resetUIState(force: Boolean = false) {
        val shouldReset = force || !model.isBenchmarkRunning()
        Log.d(TAG, "resetUIState called (force=$force, shouldReset=$shouldReset)", Exception("resetUIState"))
        if (shouldReset) {
            // Reset stopping flag when UI is reset
            isStopping = false
            
            view.setStartButtonText(context.getString(R.string.start_test))
            view.setStartButtonEnabled(true)
            view.hideProgressBar()
            view.hideStatus()
            view.hideResults()
        }
    }
    
    /**
     * Format progress message using structured data for internationalization
     */
    private fun formatProgressMessage(progress: BenchmarkProgress): BenchmarkProgress {
        val formattedMessage = when (progress.progressType) {
            ProgressType.INITIALIZING -> {
                context.getString(R.string.benchmark_initializing)
            }
            ProgressType.WARMING_UP -> {
                context.getString(R.string.benchmark_warming_up)
            }
            ProgressType.RUNNING_TEST -> {
                if (progress.runTimeSeconds > 0.0f) {
                    // Detailed progress with timing information
                    context.getString(
                        R.string.benchmark_progress_detailed,
                        progress.currentIteration,
                        progress.totalIterations,
                        progress.nPrompt,
                        progress.nGenerate,
                        progress.runTimeSeconds,
                        progress.prefillTimeSeconds,
                        progress.decodeTimeSeconds,
                        progress.prefillSpeed,
                        progress.decodeSpeed
                    )
                } else {
                    // Simple progress without timing
                    context.getString(
                        R.string.benchmark_progress_simple,
                        progress.currentIteration,
                        progress.totalIterations,
                        progress.nPrompt,
                        progress.nGenerate
                    )
                }
            }
            ProgressType.PROCESSING_RESULTS -> {
                context.getString(R.string.benchmark_processing_results)
            }
            ProgressType.COMPLETED -> {
                context.getString(R.string.benchmark_all_tests_completed)
            }
            ProgressType.STOPPING -> {
                context.getString(R.string.benchmark_stopping)
            }
            else -> {
                // Fallback to original message
                progress.statusMessage
            }
        }
        
        return progress.copy(statusMessage = formattedMessage)
    }
} 