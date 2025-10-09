package com.alibaba.mnnllm.android.benchmark

import android.util.Log

/**
 * Benchmark state enumeration
 */
enum class BenchmarkState {
    IDLE,                  // default Model
    LOADING_MODELS,        // Loading model list
    READY,                 // Ready to start benchmark
    INITIALIZING,          // Model initialization in progress
    RUNNING,               // Benchmark running
    STOPPING,              // User requested stop, stopping in progress
    COMPLETED,             // Benchmark completed, showing results
    ERROR_MODEL_NOT_FOUND, // Model not found
    ERROR                  // Error occurred
}

/**
 * State machine for benchmark functionality
 */
class BenchmarkStateMachine {
    companion object {
        private const val TAG = "BenchmarkStateMachine"
    }
    
    private var currentState: BenchmarkState = BenchmarkState.IDLE
    private var errorMessage: String? = null
    
    init {
        Log.d(TAG, "StateMachine initialized with state: $currentState")
    }
    
    fun getCurrentState(): BenchmarkState = currentState
    
    fun getErrorMessage(): String? = errorMessage
    
    /**
     * Transition to new state with validation
     */
    fun transitionTo(newState: BenchmarkState, error: String? = null) {
        val oldState = currentState
        Log.d(TAG, "Attempting transition: $oldState -> $newState${if (error != null) " (error: $error)" else ""}")
        
        // Allow self-transitions (staying in the same state)
        if (currentState == newState) {
            Log.d(TAG, "Self-transition to $newState, updating error message only")
            errorMessage = error
            return
        }
        
        if (isValidTransition(currentState, newState)) {
            currentState = newState
            errorMessage = error
            Log.d(TAG, "✓ State transition successful: $oldState -> $newState")
        } else {
            Log.e(TAG, "✗ Invalid state transition from $currentState to $newState")
            throw IllegalStateException("Invalid state transition from $currentState to $newState")
        }
    }

    fun isValidTransition(to: BenchmarkState): Boolean {
        return isValidTransition(currentState, to)
    }
    /**
     * Check if transition is valid
     */
    fun isValidTransition(from: BenchmarkState, to: BenchmarkState): Boolean {
        val validTransitions = when (from) {
            BenchmarkState.IDLE -> listOf(BenchmarkState.LOADING_MODELS)
            BenchmarkState.LOADING_MODELS -> listOf(BenchmarkState.READY, BenchmarkState.ERROR_MODEL_NOT_FOUND, BenchmarkState.ERROR)
            BenchmarkState.READY -> listOf(BenchmarkState.INITIALIZING, BenchmarkState.LOADING_MODELS)
            BenchmarkState.INITIALIZING -> listOf(BenchmarkState.RUNNING, BenchmarkState.ERROR)
            BenchmarkState.RUNNING -> listOf(BenchmarkState.STOPPING, BenchmarkState.COMPLETED, BenchmarkState.ERROR)
            BenchmarkState.STOPPING -> listOf(BenchmarkState.READY, BenchmarkState.ERROR)
            BenchmarkState.COMPLETED -> listOf(BenchmarkState.READY, BenchmarkState.INITIALIZING, BenchmarkState.LOADING_MODELS)
            BenchmarkState.ERROR -> listOf(BenchmarkState.READY, BenchmarkState.LOADING_MODELS)
            BenchmarkState.ERROR_MODEL_NOT_FOUND -> listOf(BenchmarkState.LOADING_MODELS)
        }
        
        val isValid = to in validTransitions
        Log.v(TAG, "Transition validation: $from -> $to = $isValid (valid: $validTransitions)")
        return isValid
    }
    
    /**
     * Check if benchmark can be started
     */
    fun canStart(): Boolean {
        val canStart = currentState == BenchmarkState.READY || currentState == BenchmarkState.COMPLETED
        Log.v(TAG, "canStart: $canStart (current state: $currentState)")
        return canStart
    }
    
    /**
     * Check if benchmark can be stopped
     */
    fun canStop(): Boolean {
        val canStop = currentState in listOf(BenchmarkState.RUNNING, BenchmarkState.INITIALIZING)
        Log.v(TAG, "canStop: $canStop (current state: $currentState)")
        return canStop
    }
    
    /**
     * Check if results can be shown
     */
    fun canShowResults(): Boolean = currentState == BenchmarkState.COMPLETED
    
    /**
     * Check if UI should accept user input
     */
    fun shouldAcceptInput(): Boolean = currentState in listOf(BenchmarkState.READY, BenchmarkState.COMPLETED)
    
    /**
     * Check if we're already in the target state
     */
    fun isInState(state: BenchmarkState): Boolean = currentState == state
    
    /**
     * Transition to new state only if not already in that state
     */
    fun ensureInState(state: BenchmarkState, error: String? = null) {
        Log.d(TAG, "Ensuring state: $state (current: $currentState)")
        if (!isInState(state)) {
            Log.d(TAG, "State change needed, transitioning to $state")
            transitionTo(state, error)
        } else {
            Log.d(TAG, "Already in state $state, no transition needed")
            if (error != null) {
                Log.d(TAG, "Updating error message: $error")
                errorMessage = error
            }
        }
    }
}

/**
 * UI state configuration for each benchmark state
 */
data class BenchmarkUIState(
    val startButtonText: String,
    val startButtonEnabled: Boolean,
    val showProgressBar: Boolean,
    val showResults: Boolean,
    val showStatus: Boolean,
    val statusMessage: String? = null,
    val enableModelSelector: Boolean = true,
    val showBenchmarkIcon: Boolean = true,
    val showBenchmarkProgressBar: Boolean = false,
    val benchmarkProgress: Int = 0,
    val showBackButton: Boolean = false,
    val showModelSelectorCard: Boolean = true,
    val showProgressCard: Boolean = false,
    val showStatusCard: Boolean = false
) 