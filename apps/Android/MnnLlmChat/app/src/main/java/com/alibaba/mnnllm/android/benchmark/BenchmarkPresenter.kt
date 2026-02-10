package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.Log
import androidx.lifecycle.LifecycleCoroutineScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelist.ModelListManager
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first

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
    private var selectedModelWrapper: ModelItemWrapper? = null
    private var availableModels: List<ModelItemWrapper> = emptyList()
    private val stateMachine = BenchmarkStateMachine()
    private val leaderboardService = LeaderboardService()
    private var currentBenchmarkResults: BenchmarkContract.BenchmarkResults? = null
    
    // Feature toggle: true for leaderboard upload, false for share
    // You can change this to false to test the share functionality
    private val useLeaderboardUpload = false
    
    override fun onCreate() {
        Log.d(TAG, "onCreate called, initial state: ${stateMachine.getCurrentState()}")
        setupModelSelector()
    }
    
    /**
     * Update UI based on current state
     */
    private fun updateUIForState(state: BenchmarkState) {
        Log.d(TAG, "Updating UI for state: $state")
        val uiState = when (state) {
            BenchmarkState.IDLE -> BenchmarkUIState(
                startButtonText = context.getString(R.string.start_test),
                startButtonEnabled = false,
                showProgressBar = true,
                showResults = false,
                showStatus = true,
                statusMessage = "Loading models...",
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = false,
                showModelSelectorCard = true, // Show model selector card (like iOS)
                showProgressCard = false,
                showStatusCard = true // Show status card for loading message
            )
            BenchmarkState.LOADING_MODELS -> BenchmarkUIState(
                startButtonText = context.getString(R.string.start_test),
                startButtonEnabled = false,
                showProgressBar = true,
                showResults = false,
                showStatus = true,
                statusMessage = "Loading models...",
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = false,
                showModelSelectorCard = true, // Show model selector card (like iOS)
                showProgressCard = false,
                showStatusCard = false // Show status card for loading message
            )
            BenchmarkState.READY -> BenchmarkUIState(
                startButtonText = context.getString(R.string.start_test),
                startButtonEnabled = true,
                showProgressBar = false,
                showResults = false,
                showStatus = true,
                statusMessage = context.getString(R.string.select_a_model_to_start),
                enableModelSelector = true,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = false,
                showModelSelectorCard = true, // Show model selector card (like iOS)
                showProgressCard = false,
                showStatusCard = false // Show status card for instructions
            )
            BenchmarkState.INITIALIZING -> BenchmarkUIState(
                startButtonText = context.getString(R.string.stop_test),
                startButtonEnabled = true,
                showProgressBar = true,
                showResults = false,
                showStatus = true,
                statusMessage = context.getString(R.string.benchmark_loading_model),
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = true,
                benchmarkProgress = 0, // Initial 0% before initialization
                showProgressCard = true,
                showStatusCard = true,
                showModelSelectorCard = true // Show model selector card (like iOS)
            )
            BenchmarkState.RUNNING -> BenchmarkUIState(
                startButtonText = context.getString(R.string.stop_test),
                startButtonEnabled = true,
                showProgressBar = true,
                showResults = false,
                showStatus = true,
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = true,
                benchmarkProgress = 10, // Initial 10% for entering running state
                showProgressCard = true,
                showStatusCard = true,
                showModelSelectorCard = true // Show model selector card (like iOS)
            )
            BenchmarkState.STOPPING -> BenchmarkUIState(
                startButtonText = context.getString(R.string.stop_test),
                startButtonEnabled = false,
                showProgressBar = true,
                showResults = false,
                showStatus = true,
                statusMessage = context.getString(R.string.benchmark_stopping),
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = true,
                showProgressCard = true,
                showStatusCard = true,
                showModelSelectorCard = true // Show model selector card (like iOS)
            )
            BenchmarkState.COMPLETED -> BenchmarkUIState(
                startButtonText = context.getString(R.string.restart_test), //Changed to "Re-evaluate"
                startButtonEnabled = true,
                showProgressBar = false,
                showResults = true,
                showStatus = false,
                enableModelSelector = true, // Enable model selector in results view (like iOS)
                showBenchmarkIcon = false, // Hide icon when showing results
                showBenchmarkProgressBar = false,
                showBackButton = false, // Back button removed, share button in result card instead
                showModelSelectorCard = true, // Show model selector card in results view (like iOS)
                showProgressCard = false,
                showStatusCard = false // Hide status card when showing results
            )
            BenchmarkState.ERROR -> BenchmarkUIState(
                startButtonText = context.getString(R.string.start_test),
                startButtonEnabled = true,
                showProgressBar = false,
                showResults = false,
                showStatus = false,
                enableModelSelector = true,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = false,
                showModelSelectorCard = true, // Show model selector card (like iOS)
                showProgressCard = false,
                showStatusCard = false // Hide status card for error state
            )
            BenchmarkState.ERROR_MODEL_NOT_FOUND -> BenchmarkUIState(
                startButtonText = context.getString(R.string.start_test),
                startButtonEnabled = false,
                showProgressBar = false,
                showResults = false,
                showStatus = true,
                enableModelSelector = false,
                showBenchmarkIcon = true,
                showBenchmarkProgressBar = false,
                statusMessage = context.getString(R.string.no_models_found),
                showModelSelectorCard = true, // Show model selector card (like iOS)
                showProgressCard = false,
                showStatusCard = true // Show status card for error message
            )
        }
        
        applyUIState(uiState)
    }
    
    /**
     * Apply UI state to view
     */
    private fun applyUIState(uiState: BenchmarkUIState) {
        Log.d(TAG, "Applying UI state: buttonText='${uiState.startButtonText}', buttonEnabled=${uiState.startButtonEnabled}, showProgressBar=${uiState.showProgressBar}, showResults=${uiState.showResults}, showStatus=${uiState.showStatus}, showBenchmarkIcon=${uiState.showBenchmarkIcon}, showBenchmarkProgressBar=${uiState.showBenchmarkProgressBar}, benchmarkProgress=${uiState.benchmarkProgress}, showBackButton=${uiState.showBackButton}, showModelSelectorCard=${uiState.showModelSelectorCard}, showProgressCard=${uiState.showProgressCard}, showStatusCard=${uiState.showStatusCard}")
        
        view.setStartButtonText(uiState.startButtonText)
        view.setStartButtonEnabled(uiState.startButtonEnabled)
        
        if (uiState.showProgressBar) {
            view.showProgressBar()
        } else {
            view.hideProgressBar()
        }
        
        if (uiState.showResults) {
            // Results are shown when explicitly called
        } else {
            view.hideResults()
        }
        
        if (uiState.showStatus && uiState.statusMessage != null) {
            Log.d(TAG, "Showing status: ${uiState.statusMessage}")
            view.updateStatus(uiState.statusMessage)
        } else {
            view.hideStatus()
        }
        
        // Apply new UI controls
        view.showBenchmarkIcon(uiState.showBenchmarkIcon)
        view.showBenchmarkProgressBar(uiState.showBenchmarkProgressBar)
        if (uiState.showBenchmarkProgressBar) {
            view.updateBenchmarkProgress(uiState.benchmarkProgress)
        }
        
        // Model selector enable/disable logic
        view.enableModelSelector(uiState.enableModelSelector)
        
        // Apply new button layout controls
        view.updateButtonLayout(uiState.showBackButton)
        view.showModelSelectorCard(uiState.showModelSelectorCard)
        
        // Apply progress and status cards
        view.showProgressCard(uiState.showProgressCard)
        view.showStatusCard(uiState.showStatusCard)
    }
    
    override fun onDestroy() {
        Log.d(TAG, "onDestroy called, final state: ${stateMachine.getCurrentState()}")
        model.release()
    }
    
    override fun onStartBenchmarkClicked() {
        val currentState = stateMachine.getCurrentState()
        Log.d(TAG, "onStartBenchmarkClicked called, state: $currentState")
        Log.d(TAG, "canStart: ${stateMachine.canStart()}, canStop: ${stateMachine.canStop()}")
        
        when (currentState) {
            BenchmarkState.READY -> {
                Log.d(TAG, "In READY state, checking if can start")
                if (stateMachine.canStart()) {
                    Log.d(TAG, "Starting benchmark...")
                    startBenchmark()
                } else {
                    Log.w(TAG, "Cannot start benchmark in state: $currentState")
                }
            }
            BenchmarkState.COMPLETED -> {
                Log.d(TAG, "In COMPLETED state, restarting benchmark")
                // Restart benchmark instead of sharing
                if (stateMachine.canStart()) {
                    Log.d(TAG, "Restarting benchmark...")
                    startBenchmark()
                } else {
                    Log.w(TAG, "Cannot restart benchmark in state: $currentState")
                }
            }
            BenchmarkState.RUNNING, BenchmarkState.INITIALIZING -> {
                Log.d(TAG, "In RUNNING/INITIALIZING state, checking if can stop")
                if (stateMachine.canStop()) {
                    Log.d(TAG, "Showing stop confirmation dialog")
                    view.showStopConfirmationDialog()
                } else {
                    Log.w(TAG, "Cannot stop benchmark in state: $currentState")
                }
            }
            else -> {
                Log.w(TAG, "Button click ignored in state: $currentState")
            }
        }
    }
    
    override fun onStopBenchmarkClicked() {
        val currentState = stateMachine.getCurrentState()
        Log.d(TAG, "onStopBenchmarkClicked called, state: $currentState")
        Log.d(TAG, "canStop: ${stateMachine.canStop()}")
        
        if (stateMachine.canStop()) {
            Log.d(TAG, "Stopping benchmark...")
            stopBenchmark()
        } else {
            Log.w(TAG, "Cannot stop benchmark in state: $currentState")
        }
    }
    
    override fun onModelSelected(modelWrapper: ModelItemWrapper) {
        Log.d(TAG, "onModelSelected called, state: ${stateMachine.getCurrentState()}, model: ${modelWrapper.displayName}")
        selectedModelWrapper = modelWrapper
        view.setSelectedModel(modelWrapper)
        Log.d(TAG, "Selected model: ${modelWrapper.displayName}, isLocal: ${modelWrapper.modelItem.isLocal}, localPath: ${modelWrapper.modelItem.localPath}")
    }
    
    override fun onDeleteResultClicked() {
        val currentState = stateMachine.getCurrentState()
        Log.d(TAG, "onDeleteResultClicked called, state: $currentState")
        
        if (currentState == BenchmarkState.COMPLETED) {
            Log.d(TAG, "Deleting results and transitioning to READY state")
            stateMachine.transitionTo(BenchmarkState.READY)
            updateUIForState(BenchmarkState.READY)
            view.showToast("Result deleted")
        } else {
            Log.w(TAG, "Cannot delete results in state: $currentState")
        }
    }

    override fun onBackClicked() {
        val currentState = stateMachine.getCurrentState()
        Log.d(TAG, "onBackClicked called, state: $currentState")
        
        if (currentState == BenchmarkState.COMPLETED) {
            Log.d(TAG, "Back button clicked from results, transitioning to READY state")
            stateMachine.transitionTo(BenchmarkState.READY)
            updateUIForState(BenchmarkState.READY)
        } else {
            Log.w(TAG, "Back button clicked in unexpected state: $currentState")
        }
    }
    
    override fun onSubmitResultClicked() {
    }
    
    override fun onViewLeaderboardClicked() {
    }
    
    override fun onUploadToLeaderboardClicked() {
        val results = currentBenchmarkResults
        if (results == null) {
            Log.w(TAG, "No benchmark results available for upload")
            view.showError("No benchmark results available")
            return
        }
        
        Log.d(TAG, "Starting leaderboard upload")
        
        lifecycleScope.launch {
            try {
                view.showUploadProgress("Uploading results to leaderboard...")
                
                // Extract speeds from test results
                val statistics = BenchmarkResultsHelper.processTestResults(context, results.testResults)
                val prefillSpeed = statistics.prefillStats?.average ?: 0.0
                val decodeSpeed = statistics.decodeStats?.average ?: 0.0
                val memoryUsageMb = results.maxMemoryKb / 1024.0
                
                // Submit score to leaderboard
                val submitResult = leaderboardService.submitScore(
                    context,
                    results.modelDisplayName,
                    prefillSpeed,
                    decodeSpeed,
                    memoryUsageMb
                )
                
                when (submitResult) {
                    is LeaderboardService.SubmitResult.Success -> {
                        Log.d(TAG, "Upload successful, getting rank")
                        view.showUploadProgress("Getting your ranking...")
                        
                        // Get user ranking
                        val rankResult = leaderboardService.getUserRank(
                            context,
                            results.modelDisplayName
                        )
                        
                        view.hideUploadProgress()
                        
                        when (rankResult) {
                            is LeaderboardService.RankResult.Success -> {
                                val rankData = rankResult.rankData
                                view.showRankInfo(rankData.rank, rankData.totalUsers)
                                view.showToast("Successfully uploaded to leaderboard!")
                            }
                            is LeaderboardService.RankResult.Error -> {
                                Log.w(TAG, "Failed to get rank: ${rankResult.message}")
                                view.showToast("Uploaded successfully, but couldn't get ranking")
                            }
                        }
                    }
                    is LeaderboardService.SubmitResult.Error -> {
                        Log.e(TAG, "Upload failed: ${submitResult.message}")
                        view.hideUploadProgress()
                        view.showError("Upload failed: ${submitResult.message}")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Upload error", e)
                view.hideUploadProgress()
                view.showError("Upload failed: ${e.message}")
            }
        }
    }
    
    override fun getCurrentState(): BenchmarkState {
        return stateMachine.getCurrentState()
    }
    
    /**
     * Setup model selector with available models
     */
    private fun setupModelSelector() {
        Log.d(TAG, "setupModelSelector called")
        // Ensure we're in LOADING_MODELS state
        if (!stateMachine.isValidTransition(BenchmarkState.LOADING_MODELS)) {
            return
        }
        stateMachine.ensureInState(BenchmarkState.LOADING_MODELS)
        updateUIForState(BenchmarkState.LOADING_MODELS)
        
        lifecycleScope.launch {
            try {
                Log.d(TAG, "Loading available models...")
                Log.d(TAG, "Calling ModelListManager.initialize from BenchmarkPresenter", Throwable())
                // Get current models or wait for them
                val models = ModelListManager.getCurrentModels()?: emptyList()
                availableModels = models.filterNot { ModelTypeUtils.isDiffusionModel(
                    it.modelItem.modelName ?: ""
                ) }
                Log.d(TAG, "Found ${availableModels.size} models")
                view.updateModelSelector(availableModels)
                if (availableModels.isEmpty()) {
                    Log.e(TAG, "No models available")
                    stateMachine.transitionTo(BenchmarkState.ERROR, context.getString(R.string.no_models_available))
                    updateUIForState(BenchmarkState.ERROR)
                } else {
                    // Set first model as default
                    selectedModelWrapper = availableModels.firstOrNull()
                    selectedModelWrapper?.let { wrapper ->
                        Log.d(TAG, "Setting default model: ${wrapper.displayName}")
                        view.setSelectedModel(wrapper)
                    }
                    
                    // Transition to READY state
                    Log.d(TAG, "Models loaded successfully, transitioning to READY state")
                    stateMachine.transitionTo(BenchmarkState.READY)
                    updateUIForState(BenchmarkState.READY)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load models", e)
                stateMachine.transitionTo(BenchmarkState.ERROR_MODEL_NOT_FOUND, "Failed to load models: ${e.message}")
//                view.showError("Failed to load models: ${e.message}")
                updateUIForState(BenchmarkState.ERROR_MODEL_NOT_FOUND)
            }
        }
    }
    
    /**
     * Start benchmark test
     */
    private fun startBenchmark() {
        Log.d(TAG, "startBenchmark called, current state: ${stateMachine.getCurrentState()}")
        val modelWrapper = selectedModelWrapper
        if (modelWrapper == null) {
            Log.e(TAG, "No model selected")
            view.showError("Please select a model")
            return
        }
        
        // Get selected backend
        val backendType = view.getSelectedBackend()
        Log.d(TAG, "Selected backend: $backendType")
        
        Log.d(TAG, "Starting benchmark with model: ${modelWrapper.displayName}")
        
        // Transition to INITIALIZING state
        Log.d(TAG, "Transitioning to INITIALIZING state")
        stateMachine.transitionTo(BenchmarkState.INITIALIZING)
        updateUIForState(BenchmarkState.INITIALIZING)
        
        lifecycleScope.launch {
            try {
                // Initialize model if not already done or backend changed
                if (!model.isModelInitialized() || 
                    model.getModelInfo() != modelWrapper.modelItem.modelId ||
                    model.getBackendType() != backendType) {
                    
                    Log.d(TAG, "Model needs initialization or re-initialization (backend changed)")
                    
                    val configPath = if (modelWrapper.modelItem.isLocal && !modelWrapper.modelItem.localPath.isNullOrEmpty()) {
                        "${modelWrapper.modelItem.localPath}/config.json"
                    } else {
                        null
                    }
                    
                    Log.d(TAG, "Initializing model with config: $configPath, backend: $backendType")
                    val initSuccess = model.initializeModel(modelWrapper.modelItem.modelId!!, configPath, backendType)
                    
                    if (!initSuccess) {
                        Log.e(TAG, "Model initialization failed")
                        stateMachine.transitionTo(BenchmarkState.ERROR, "Failed to initialize model")
                        view.showError("Failed to initialize model")
                        updateUIForState(BenchmarkState.ERROR)
                        return@launch
                    }
                    Log.d(TAG, "Model initialization successful")
                } else {
                    Log.d(TAG, "Model already initialized, skipping init")
                }
                
                // Transition to RUNNING state
                stateMachine.transitionTo(BenchmarkState.RUNNING)
                // Create custom UI state for RUNNING with 10% progress (5% start + 5% initialization)
                val runningUIState = BenchmarkUIState(
                    startButtonText = context.getString(R.string.stop_test),
                    startButtonEnabled = true,
                    showProgressBar = true,
                    showResults = false,
                    showStatus = true,
                    enableModelSelector = false,
                    showBenchmarkIcon = true,
                    showBenchmarkProgressBar = true,
                    benchmarkProgress = 10, // 10% for entering running state
                    showModelSelectorCard = true,
                    showProgressCard = true, //Critical fix: show progress card
                    showStatusCard = true //Critical fix: show status card
                )
                applyUIState(runningUIState)
                
                // Start benchmark
                model.startBenchmark(
                    context,
                    modelWrapper,
                    object : BenchmarkModel.BenchmarkModelCallback {
                        override fun onProgress(progress: BenchmarkProgress) {
                            val currentState = stateMachine.getCurrentState()
                            Log.v(TAG, "onProgress called, state: $currentState, progress: ${progress.progress}%")
                            
                            // Only process progress updates if in RUNNING state
                            if (currentState != BenchmarkState.RUNNING) {
                                Log.d(TAG, "Ignoring progress update in state: $currentState")
                                return
                            }
                            
                            // Calculate real progress based on token processing
                            val realProgress = calculateRealProgress(progress)
                            Log.d(TAG, "onProgress: calculated realProgress=$realProgress for progressType=${progress.progressType}, nativeProgress=${progress.progress}")
                            
                            // Update UI with real progress
                            val uiState = when (currentState) {
                                BenchmarkState.RUNNING -> BenchmarkUIState(
                                    startButtonText = context.getString(R.string.stop_test),
                                    startButtonEnabled = true,
                                    showProgressBar = true,
                                    showResults = false,
                                    showStatus = true,
                                    enableModelSelector = false,
                                    showBenchmarkIcon = true,
                                    showBenchmarkProgressBar = true,
                                    benchmarkProgress = realProgress,
                                    showModelSelectorCard = true,
                                    showProgressCard = true, //Critical fix: show progress card
                                    showStatusCard = true //Critical fix: show status card
                                )
                                else -> return
                            }
                            applyUIState(uiState)
                            
                            // Format progress message based on structured data
                            val formattedProgress = formatProgressMessage(progress)
                            view.updateProgress(formattedProgress)
                            
                            // Update progress card with detailed information - use realProgress instead of native progress
                            // Note: realProgress is already applied via UI state, no need to call again here
                            if (formattedProgress.statusMessage.isNotEmpty()) {
                                view.updateStatusMessage(formattedProgress.statusMessage)
                            }
                            
                            // Update test details if available
                            if (progress.totalIterations > 0) {
                                view.updateTestDetails(
                                    progress.currentIteration,
                                    progress.totalIterations,
                                    progress.nPrompt,
                                    progress.nGenerate
                                )
                            }
                            
                            // Update performance metrics if available
                            if (progress.runTimeSeconds > 0) {
                                view.updateProgressMetrics(
                                    progress.runTimeSeconds,
                                    progress.prefillTimeSeconds,
                                    progress.decodeTimeSeconds,
                                    progress.prefillSpeed,
                                    progress.decodeSpeed
                                )
                            }
                            
                            Log.d(TAG, "Benchmark Progress (${progress.progress}% -> ${realProgress}% real): ${formattedProgress.statusMessage}")
                        }
                        
                        override fun onComplete(results: BenchmarkContract.BenchmarkResults) {
                            val currentState = stateMachine.getCurrentState()
                            Log.d(TAG, "onComplete called, state: $currentState, model: ${results.modelDisplayName}")
                            
                            // Only process completion if in RUNNING state
                            if (currentState != BenchmarkState.RUNNING) {
                                Log.d(TAG, "Ignoring completion results in state: $currentState")
                                return
                            }
                            
                            // Save results for leaderboard upload
                            currentBenchmarkResults = results
                            
                            // Transition to COMPLETED state and show results
                            Log.d(TAG, "Transitioning to COMPLETED state and showing results")
                            stateMachine.transitionTo(BenchmarkState.COMPLETED)
                            view.showResults(results)
                            updateUIForState(BenchmarkState.COMPLETED)
                        }
                        
                        override fun onBenchmarkModelError(errorCode: Int, message: String) {
                            val currentState = stateMachine.getCurrentState()
                            Log.e(TAG, "onBenchmarkModelError called, state: $currentState, errorCode: $errorCode, error: $message")
                            
                            // Check if this is a user-initiated stop
                            if (errorCode == BenchmarkErrorCode.BENCHMARK_STOPPED) {
                                Log.d(TAG, "Benchmark stopped by user (errorCode: $errorCode), transitioning to READY state")
                                stateMachine.transitionTo(BenchmarkState.READY)
                                updateUIForState(BenchmarkState.READY)
                            } else {
                                Log.d(TAG, "Transitioning to ERROR state due to benchmark error (errorCode: $errorCode)")
                                stateMachine.transitionTo(BenchmarkState.ERROR, "Benchmark failed: $message")
                                view.showError("Benchmark failed: $message")
                                updateUIForState(BenchmarkState.ERROR)
                            }
                        }
                    },
                    backendType
                )
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark start failed", e)
                stateMachine.transitionTo(BenchmarkState.ERROR, "Error: ${e.message}")
                view.showError("Error: ${e.message}")
                updateUIForState(BenchmarkState.ERROR)
            }
        }
    }
    
    /**
     * Stop benchmark test
     */
    private fun stopBenchmark() {
        Log.d(TAG, "stopBenchmark called, current state: ${stateMachine.getCurrentState()}")
        
        // Transition to STOPPING state
        Log.d(TAG, "Transitioning to STOPPING state")
        if (!stateMachine.isValidTransition(BenchmarkState.STOPPING)) {
            Log.e(TAG, "cannot stop in current  state: ${stateMachine.getCurrentState()}")
            return
        }
        stateMachine.transitionTo(BenchmarkState.STOPPING)
        updateUIForState(BenchmarkState.STOPPING)
        
        // Stop the benchmark - the actual state transition will happen in callback
        Log.d(TAG, "Calling model.stopBenchmark(), waiting for callback to complete state transition")
        model.stopBenchmark()
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
    
    /**
     * Calculate real progress based on benchmark state
     * - Running state start: 10%
     * - After warming up: at least 20%
     * - Remaining realProgress distributed over remaining 80%
     */
    private fun calculateRealProgress(progress: BenchmarkProgress): Int {
        Log.d(TAG, "calculateRealProgress: progressType=${progress.progressType}, nativeProgress=${progress.progress}, currentIteration=${progress.currentIteration}, totalIterations=${progress.totalIterations}")
        
        // Base progress for entering RUNNING state: 10%
        val runningStateStart = 10
        
        // After warming up: at least 20%
        val afterWarmupMin = 20
        
        // Remaining 80% for actual progress distribution
        val remainingProgressRange = 80
        
        // Calculate progress based on progressType
        val finalProgress = when (progress.progressType) {
            ProgressType.INITIALIZING -> {
                // During initialization: 0-10%
                val initProgress = (progress.progress.coerceIn(0, 100) / 100.0f * runningStateStart).toInt()
                initProgress.coerceIn(0, runningStateStart)
            }
            ProgressType.WARMING_UP -> {
                // During warming up: 10-20%
                val warmupProgress = runningStateStart + (progress.progress.coerceIn(0, 100) / 100.0f * (afterWarmupMin - runningStateStart)).toInt()
                warmupProgress.coerceIn(runningStateStart, afterWarmupMin)
            }
            ProgressType.RUNNING_TEST -> {
                // After warming up: 20-100%, distributed over remaining 80%
                
                // If we have iteration information, calculate based on that
                if (progress.totalIterations > 0 && progress.currentIteration >= 0) {
                    val iterationProgress = (progress.currentIteration.toFloat() / progress.totalIterations.toFloat() * remainingProgressRange).toInt()
                    (afterWarmupMin + iterationProgress).coerceIn(afterWarmupMin, 100)
                }
                // If we have token information, calculate based on tokens
                else if (progress.nPrompt > 0 && progress.nGenerate > 0) {
                    // Use the native progress percentage if available, but scale it to our remaining 80% range
                    val nativeProgress = progress.progress.coerceIn(0, 100)
                    val scaledProgress = (nativeProgress / 100.0f * remainingProgressRange).toInt()
                    
                    (afterWarmupMin + scaledProgress).coerceIn(afterWarmupMin, 100)
                }
                // Fallback: use native progress but scale to remaining 80%
                else {
                    val fallbackProgress = progress.progress.coerceIn(0, 100)
                    val scaledFallback = (fallbackProgress / 100.0f * remainingProgressRange).toInt()
                    
                    (afterWarmupMin + scaledFallback).coerceIn(afterWarmupMin, 100)
                }
            }
            else -> {
                // Fallback for unknown states
                runningStateStart
            }
        }
        
        Log.d(TAG, "calculateRealProgress result: $finalProgress")
        return finalProgress
    }

    /**
     * Public method to reload models, for fragment re-show
     */
    fun loadModels() {
        setupModelSelector()
    }
} 