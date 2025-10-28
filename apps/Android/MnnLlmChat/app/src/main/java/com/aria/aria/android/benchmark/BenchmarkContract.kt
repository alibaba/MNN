package com.alibaba.mnnllm.android.benchmark

import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelist.ModelListManager

/**
 * Contract between BenchmarkView and BenchmarkPresenter
 */
class BenchmarkContract {
    
    interface View {
        fun showLoading()
        fun hideLoading()
        fun showError(message: String)
        fun showToast(message: String)
        fun showStopConfirmationDialog()
        
        // Model selection
        fun updateModelSelector(models: List<ModelItemWrapper>)
        fun setSelectedModel(modelWrapper: ModelItemWrapper)
        fun enableStartButton(enabled: Boolean)
        
        // Benchmark progress
        fun updateProgress(progress: BenchmarkProgress)
        fun showResults(results: BenchmarkResults)
        fun hideResults()
        fun updateStatus(message: String)
        fun hideStatus()
        
        // Progress and Status Cards
        fun showProgressCard(show: Boolean)
        fun showStatusCard(show: Boolean)
        fun updateStatusMessage(message: String)
        fun updateTestDetails(
            currentIteration: Int,
            totalIterations: Int,
            nPrompt: Int,
            nGenerate: Int
        )
        fun updateProgressMetrics(
            runtime: Float,
            prefillTime: Float,
            decodeTime: Float,
            prefillSpeed: Float,
            decodeSpeed: Float
        )
        
        // UI state
        fun setStartButtonText(text: String)
        fun setStartButtonEnabled(enabled: Boolean)
        fun showProgressBar()
        fun hideProgressBar()
        
        // Benchmark icon and progress
        fun showBenchmarkIcon(show: Boolean)
        fun showBenchmarkProgressBar(show: Boolean)
        fun updateBenchmarkProgress(progress: Int)
        fun enableModelSelector(enabled: Boolean)
        
        // Button layout control
        fun showBackButton(show: Boolean)
        fun showModelSelectorCard(show: Boolean)
        fun updateButtonLayout(showBackButton: Boolean)
        
        // Share and Leaderboard functionality
        fun shareResultCard()
        fun uploadToLeaderboard()
        fun showUploadProgress(message: String)
        fun hideUploadProgress()
        fun showRankInfo(rank: Int, totalUsers: Int)
    }
    
    interface Presenter {
        fun onCreate()
        fun onDestroy()
        fun onStartBenchmarkClicked()
        fun onStopBenchmarkClicked()
        fun onModelSelected(modelWrapper: ModelItemWrapper)
        fun onDeleteResultClicked()
        fun onSubmitResultClicked()
        fun onViewLeaderboardClicked()
        fun onBackClicked()
        fun onUploadToLeaderboardClicked()
        fun getCurrentState(): BenchmarkState
    }
    
    data class BenchmarkResults(
        val modelDisplayName: String,
        val maxMemoryKb: Long,
        val testResults: List<TestInstance>,
        val timestamp: String
    )
} 