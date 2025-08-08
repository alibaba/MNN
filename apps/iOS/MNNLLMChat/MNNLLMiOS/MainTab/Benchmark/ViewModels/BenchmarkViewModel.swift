//
//  BenchmarkViewModel.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import SwiftUI
import Combine

/**
 * ViewModel for managing benchmark operations including model selection, test execution,
 * progress tracking, and result management. Handles communication with BenchmarkService
 * and provides UI state management for the benchmark interface.
 */
@MainActor
class BenchmarkViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var isLoading = false
    @Published var isRunning = false
    @Published var showProgressBar = false
    @Published var showResults = false
    @Published var showError = false
    
    @Published var selectedModel: ModelInfo?
    @Published var availableModels: [ModelInfo] = []
    @Published var currentProgress: BenchmarkProgress?
    @Published var benchmarkResults: BenchmarkResults?
    @Published var errorMessage: String = ""
    @Published var statusMessage: String = ""
    
    @Published var startButtonText = String(localized: "Start Test")
    @Published var isStartButtonEnabled = true
    @Published var showStopConfirmation = false
    
    // MARK: - Private Properties
    
    private let benchmarkService = BenchmarkService.shared
    private let resultsHelper = BenchmarkResultsHelper.shared
    private var cancellables = Set<AnyCancellable>()
    
    // Model list manager for getting local models
    private let modelListManager = ModelListManager.shared
    
    // MARK: - Initialization & Setup
    
    init() {
        setupBindings()
        loadAvailableModels()
    }
    
    /// Sets up reactive bindings between service and view model
    private func setupBindings() {
        benchmarkService.$isRunning
            .receive(on: DispatchQueue.main)
            .assign(to: \.isRunning, on: self)
            .store(in: &cancellables)
        
        // Update button text based on running state
        benchmarkService.$isRunning
            .receive(on: DispatchQueue.main)
            .map { isRunning in
                isRunning ? String(localized: "Stop Test") : String(localized: "Start Test")
            }
            .assign(to: \.startButtonText, on: self)
            .store(in: &cancellables)
    }
    
    /// Loads available models from ModelListManager, filtering for downloaded models only
    private func loadAvailableModels() {
        Task {
            isLoading = true
            
            do {
                // Get all models from ModelListManager
                let allModels = try await modelListManager.loadModels()
                
                // Filter only downloaded models that are available locally
                availableModels = allModels.filter { model in
                    model.isDownloaded && model.localPath != nil
                }
                
                print("BenchmarkViewModel: Loaded \(availableModels.count) available local models")
                
            } catch {
                showErrorMessage("Failed to load models: \(error.localizedDescription)")
            }
            
            isLoading = false
        }
    }
    
    // MARK: - Public Action Handlers
    
    /// Handles start/stop benchmark button taps
    func onStartBenchmarkTapped() {
        if !isRunning {
            startBenchmark()
        } else {
            showStopConfirmationAlert()
        }
    }
    
    /// Handles benchmark stop confirmation
    func onStopBenchmarkTapped() {
        showStopConfirmation = false
        stopBenchmark()
    }
    
    /// Handles model selection from dropdown
    func onModelSelected(_ model: ModelInfo) {
        selectedModel = model
    }
    
    /// Handles result deletion and cleanup
    func onDeleteResultTapped() {
        benchmarkResults = nil
        showResults = false
        hideStatus()
        
        // Clean up resources when deleting results
        cleanupBenchmarkResources()
    }
    
    /// Placeholder for future result submission functionality
    func onSubmitResultTapped() {
        // Implementation for submitting results (if needed)
        // This could involve sharing or uploading results
    }
    
    // MARK: - Benchmark Execution
    
    /// Initiates benchmark test with selected model and configured parameters
    private func startBenchmark() {
        guard let model = selectedModel else {
            showErrorMessage("Please select a model first")
            return
        }
        
        guard model.isDownloaded else {
            showErrorMessage("Selected model is not downloaded or path is invalid")
            return
        }
        
        onBenchmarkStarted()
        
        Task {
            // Initialize model if needed
            let initialized = await benchmarkService.initializeModel(
                modelId: model.id,
                modelPath: model.localPath
            )
            
            guard initialized else {
                showErrorMessage("Failed to initialize model")
                resetUIState()
                return
            }
            
            // Start memory monitoring
            MemoryMonitor.shared.start()
            
            // Start benchmark with optimized parameters for mobile devices
            benchmarkService.runBenchmark(
                modelId: model.id,
                callback: self,
                runtimeParams: createRuntimeParameters(),
                testParams: createTestParameters()
            )
        }
    }
    
    /// Creates runtime parameters optimized for iOS devices
    private func createRuntimeParameters() -> RuntimeParameters {
        return RuntimeParameters(
            backends: [0], // CPU backend
            threads: [4], // 4 threads for most iOS devices
            useMmap: false, // Memory mapping disabled for iOS
            power: [0], // Normal power mode
            precision: [2], // Low precision for better performance
            memory: [2], // Low memory usage
            dynamicOption: [0] // No dynamic optimization
        )
    }
    
    /// Creates test parameters suitable for mobile benchmarking
    private func createTestParameters() -> TestParameters {
        return TestParameters(
            nPrompt: [256, 512], // Smaller prompt sizes for mobile
            nGenerate: [64, 128], // Smaller generation sizes
            nPrompGen: [(256, 64), (512, 128)], // Combined test cases
            nRepeat: [3], // Fewer repetitions for faster testing
            kvCache: "false", // Disable KV cache by default
            loadTime: "false"
        )
    }
    
    /// Stops the currently running benchmark
    private func stopBenchmark() {
        updateStatus("Stopping benchmark...")
        benchmarkService.stopBenchmark()
        cleanupBenchmarkResources()
    }
    
    // MARK: - UI State Management
    
    /// Updates UI state when benchmark starts
    private func onBenchmarkStarted() {
        isRunning = true
        isStartButtonEnabled = true
        startButtonText = String(localized: "Stop Test")
        showProgressBar = true
        showResults = false
        updateStatus("Initializing benchmark...")
    }
    
    /// Resets UI to initial state
    private func resetUIState() {
        isRunning = false
        isStartButtonEnabled = true
        startButtonText = String(localized: "Start Test")
        showProgressBar = false
        hideStatus()
        showResults = false
        cleanupBenchmarkResources()
    }
    
    /// Cleans up benchmark resources including memory monitoring and model
    private func cleanupBenchmarkResources() {
        MemoryMonitor.shared.stop()
        MemoryMonitor.shared.reset()
        
        // Release model to free memory
        benchmarkService.releaseModel()
    }
    
    /// Updates status message display
    private func updateStatus(_ message: String) {
        statusMessage = message
    }
    
    /// Hides status message
    private func hideStatus() {
        statusMessage = ""
    }
    
    /// Shows error message alert
    private func showErrorMessage(_ message: String) {
        errorMessage = message
        showError = true
    }
    
    /// Shows stop confirmation alert
    private func showStopConfirmationAlert() {
        showStopConfirmation = true
    }
    
    /// Formats progress messages with appropriate status text based on progress type
    private func formatProgressMessage(_ progress: BenchmarkProgress) -> BenchmarkProgress {
        let formattedMessage: String
        
        switch progress.progressType {
        case .initializing:
            formattedMessage = "Initializing benchmark..."
        case .warmingUp:
            formattedMessage = "Warming up..."
        case .runningTest:
            formattedMessage = "Running test \(progress.currentIteration)/\(progress.totalIterations)"
        case .processingResults:
            formattedMessage = "Processing results..."
        case .completed:
            formattedMessage = "All tests completed"
        case .stopping:
            formattedMessage = "Stopping benchmark..."
        default:
            formattedMessage = progress.statusMessage
        }
        
        return BenchmarkProgress(
            progress: progress.progress,
            statusMessage: formattedMessage,
            progressType: progress.progressType,
            currentIteration: progress.currentIteration,
            totalIterations: progress.totalIterations,
            nPrompt: progress.nPrompt,
            nGenerate: progress.nGenerate,
            runTimeSeconds: progress.runTimeSeconds,
            prefillTimeSeconds: progress.prefillTimeSeconds,
            decodeTimeSeconds: progress.decodeTimeSeconds,
            prefillSpeed: progress.prefillSpeed,
            decodeSpeed: progress.decodeSpeed
        )
    }
}

// MARK: - BenchmarkCallback Implementation

extension BenchmarkViewModel: BenchmarkCallback {
    
    /// Handles progress updates from benchmark service
    func onProgress(_ progress: BenchmarkProgress) {
        let formattedProgress = formatProgressMessage(progress)
        currentProgress = formattedProgress
        updateStatus(formattedProgress.statusMessage)
    }
    
    /// Handles benchmark completion with results processing
    func onComplete(_ result: BenchmarkResult) {
        guard let model = selectedModel else { return }
        
        updateStatus("Processing results...")
        
        // Create comprehensive benchmark results
        let results = BenchmarkResults(
            modelDisplayName: model.modelName,
            maxMemoryKb: MemoryMonitor.shared.getMaxMemoryKb(),
            testResults: [result.testInstance],
            timestamp: DateFormatter.benchmarkTimestamp.string(from: Date())
        )
        
        benchmarkResults = results
        showResults = true
        
        // Update UI state to reflect completion
        isRunning = false
        isStartButtonEnabled = true
        startButtonText = String(localized: "Start Test")
        showProgressBar = false
        
        // Clean up resources after benchmark completion
        cleanupBenchmarkResources()
        
        // Always hide status after processing results
        hideStatus()
        
        print("BenchmarkViewModel: Benchmark completed successfully for model: \(model.modelName)")
    }
    
    /// Handles benchmark errors with user-friendly error messages
    func onBenchmarkError(_ errorCode: Int, _ message: String) {
        let errorCodeName = BenchmarkErrorCode(rawValue: errorCode)?.description ?? "Unknown"
        
        // Check if this is a user-initiated stop - don't show error dialog
        if errorCode == BenchmarkErrorCode.benchmarkStopped.rawValue {
            print("BenchmarkViewModel: Benchmark stopped by user (\(errorCode)): \(message)")
        } else {
            showErrorMessage("Benchmark failed (\(errorCodeName)): \(message)")
            print("BenchmarkViewModel: Benchmark error (\(errorCode)): \(message)")
        }
        
        resetUIState()
    }
}

// MARK: - Memory Monitoring

/**
 * Singleton class for monitoring memory usage during benchmark execution.
 * Tracks current and peak memory consumption using system APIs.
 */
class MemoryMonitor: ObservableObject {
    
    static let shared = MemoryMonitor()
    
    @Published private(set) var currentMemoryKb: Int64 = 0
    private var maxMemoryKb: Int64 = 0
    private var isMonitoring = false
    private var monitoringTask: Task<Void, Never>?
    
    private init() {}
    
    /// Starts continuous memory monitoring
    func start() {
        guard !isMonitoring else { return }
        
        isMonitoring = true
        maxMemoryKb = 0
        
        monitoringTask = Task {
            while isMonitoring && !Task.isCancelled {
                await updateMemoryUsage()
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
            }
        }
    }
    
    /// Stops memory monitoring
    func stop() {
        isMonitoring = false
        monitoringTask?.cancel()
        monitoringTask = nil
    }
    
    /// Resets memory tracking counters
    func reset() {
        maxMemoryKb = 0
        currentMemoryKb = 0
    }
    
    /// Returns the maximum memory usage recorded during monitoring
    func getMaxMemoryKb() -> Int64 {
        return maxMemoryKb
    }
    
    /// Updates current memory usage and tracks maximum
    @MainActor
    private func updateMemoryUsage() {
        let memoryUsage = getCurrentMemoryUsage()
        currentMemoryKb = memoryUsage
        maxMemoryKb = max(maxMemoryKb, memoryUsage)
    }
    
    /// Gets current memory usage from system using mach task info
    private func getCurrentMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Int64(info.resident_size) / 1024 // Convert to KB
        } else {
            return 0
        }
    }
}

// MARK: - Extensions

/// Extension providing user-friendly descriptions for benchmark error codes
extension BenchmarkErrorCode {
    var description: String {
        switch self {
        case .benchmarkFailedUnknown:
            return "Unknown Error"
        case .testInstanceFailed:
            return "Test Failed"
        case .modelNotInitialized:
            return "Model Not Ready"
        case .benchmarkRunning:
            return "Already Running"
        case .benchmarkStopped:
            return "Stopped"
        case .nativeError:
            return "Native Error"
        case .modelError:
            return "Model Error"
        }
    }
}

/// Extension providing formatted timestamp for benchmark results
extension DateFormatter {
    static let benchmarkTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy/M/dd HH:mm:ss"
        return formatter
    }()
}
