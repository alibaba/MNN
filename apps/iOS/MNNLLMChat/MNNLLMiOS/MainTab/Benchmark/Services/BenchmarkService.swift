//
//  BenchmarkService.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import Combine

/**
 * Protocol defining callback methods for benchmark execution events.
 * Provides progress updates, completion notifications, and error handling.
 */
protocol BenchmarkCallback: AnyObject {
    func onProgress(_ progress: BenchmarkProgress)
    func onComplete(_ result: BenchmarkResult)
    func onBenchmarkError(_ errorCode: Int, _ message: String)
}

/**
 * Singleton service class responsible for managing benchmark operations.
 * Coordinates with LLMInferenceEngineWrapper to execute performance tests
 * and provides real-time progress updates through callback mechanisms.
 */
class BenchmarkService: ObservableObject {
    
    // MARK: - Singleton & Properties
    
    static let shared = BenchmarkService()
    
    @Published private(set) var isRunning = false
    private var shouldStop = false
    private var currentTask: Task<Void, Never>?
    
    // Real LLM inference engine - using actual MNN LLM wrapper
    private var llmEngine: LLMInferenceEngineWrapper?
    private var currentModelId: String?
    
    private init() {}
    
    // MARK: - Public Interface
    
    /// Initiates benchmark execution with specified parameters and callback handler
    /// - Parameters:
    ///   - modelId: Identifier for the model to benchmark
    ///   - callback: Callback handler for progress and completion events
    ///   - runtimeParams: Runtime configuration parameters
    ///   - testParams: Test scenario parameters
    func runBenchmark(
        modelId: String,
        callback: BenchmarkCallback,
        runtimeParams: RuntimeParameters = .default,
        testParams: TestParameters = .default
    ) {
        guard !isRunning else {
            callback.onBenchmarkError(BenchmarkErrorCode.benchmarkRunning.rawValue, "Benchmark is already running")
            return
        }
        
        guard let engine = llmEngine, engine.isModelReady() else {
            callback.onBenchmarkError(BenchmarkErrorCode.modelNotInitialized.rawValue, "Model is not initialized or not ready")
            return
        }
        
        isRunning = true
        shouldStop = false
        
        currentTask = Task {
            await performBenchmark(
                engine: engine,
                modelId: modelId,
                callback: callback,
                runtimeParams: runtimeParams,
                testParams: testParams
            )
        }
    }
    
    /// Stops the currently running benchmark operation
    func stopBenchmark() {
        shouldStop = true
        llmEngine?.stopBenchmark()
        currentTask?.cancel()
        isRunning = false
    }
    
    /// Checks if the model is properly initialized and ready for benchmarking
    /// - Returns: True if model is ready, false otherwise
    func isModelInitialized() -> Bool {
        return llmEngine != nil && llmEngine!.isModelReady()
    }
    
    /// Initializes a model for benchmark testing
    /// - Parameters:
    ///   - modelId: Identifier for the model
    ///   - modelPath: File system path to the model
    /// - Returns: True if initialization succeeded, false otherwise
    func initializeModel(modelId: String, modelPath: String) async -> Bool {
        return await withCheckedContinuation { continuation in
            // Release existing engine if any
            llmEngine = nil
            currentModelId = nil
            
            // Create new LLM inference engine
            llmEngine = LLMInferenceEngineWrapper(modelPath: modelPath) { success in
                if success {
                    self.currentModelId = modelId
                    print("BenchmarkService: Model \(modelId) initialized successfully")
                } else {
                    self.llmEngine = nil
                    print("BenchmarkService: Failed to initialize model \(modelId)")
                }
                continuation.resume(returning: success)
            }
        }
    }
    
    /// Retrieves information about the currently loaded model
    /// - Returns: Model information string, or nil if no model is loaded
    func getModelInfo() -> String? {
        guard let modelId = currentModelId else { return nil }
        return "Model: \(modelId), Engine: MNN LLM"
    }
    
    /// Releases the current model and frees associated resources
    func releaseModel() {
        llmEngine?.cancelInference()
        llmEngine = nil
        currentModelId = nil
    }
    
    // MARK: - Benchmark Execution
    
    /// Performs the actual benchmark execution with progress tracking
    /// - Parameters:
    ///   - engine: LLM inference engine instance
    ///   - modelId: Model identifier
    ///   - callback: Progress and completion callback handler
    ///   - runtimeParams: Runtime configuration
    ///   - testParams: Test parameters
    private func performBenchmark(
        engine: LLMInferenceEngineWrapper,
        modelId: String,
        callback: BenchmarkCallback,
        runtimeParams: RuntimeParameters,
        testParams: TestParameters
    ) async {
        do {
            let instances = generateTestInstances(runtimeParams: runtimeParams, testParams: testParams)
            
            var completedInstances = 0
            let totalInstances = instances.count
            
            for instance in instances {
                if shouldStop {
                    await MainActor.run {
                        callback.onBenchmarkError(BenchmarkErrorCode.benchmarkStopped.rawValue, "Benchmark stopped by user")
                        self.isRunning = false
                    }
                    return
                }
                
                // Create TestInstance for current configuration
                let testInstance = TestInstance(
                    modelConfigFile: instance.configPath,
                    modelType: modelId,
                    modelSize: 0, // Will be calculated if needed
                    threads: instance.threads,
                    useMmap: instance.useMmap,
                    nPrompt: instance.nPrompt,
                    nGenerate: instance.nGenerate,
                    backend: instance.backend,
                    precision: instance.precision,
                    power: instance.power,
                    memory: instance.memory,
                    dynamicOption: instance.dynamicOption
                )
                
                // Update overall progress
                let progress = (completedInstances * 100) / totalInstances
                let statusMsg = "Running test \(completedInstances + 1)/\(totalInstances): pp\(instance.nPrompt)+tg\(instance.nGenerate)"
                
                await MainActor.run {
                    callback.onProgress(BenchmarkProgress(
                        progress: progress,
                        statusMessage: statusMsg,
                        progressType: .runningTest,
                        currentIteration: completedInstances + 1,
                        totalIterations: totalInstances,
                        nPrompt: instance.nPrompt,
                        nGenerate: instance.nGenerate
                    ))
                }
                
                // Execute benchmark using LLMInferenceEngineWrapper
                let result = await runOfficialBenchmark(
                    engine: engine,
                    instance: instance,
                    testInstance: testInstance,
                    progressCallback: { progress in
                        await MainActor.run {
                            callback.onProgress(progress)
                        }
                    }
                )
                
                if result.success {
                    completedInstances += 1
                    
                    // Only call onComplete for the last test instance
                    if completedInstances == totalInstances {
                        await MainActor.run {
                            callback.onComplete(result)
                        }
                    }
                } else {
                    await MainActor.run {
                        callback.onBenchmarkError(BenchmarkErrorCode.testInstanceFailed.rawValue, result.errorMessage ?? "Test failed")
                        self.isRunning = false
                    }
                    return
                }
            }
            
            await MainActor.run {
                self.isRunning = false
            }
            
        } catch {
            await MainActor.run {
                callback.onBenchmarkError(BenchmarkErrorCode.nativeError.rawValue, error.localizedDescription)
                self.isRunning = false
            }
        }
    }
    
    /// Executes a single benchmark test using the official MNN LLM benchmark interface
    /// - Parameters:
    ///   - engine: LLM inference engine
    ///   - instance: Test configuration
    ///   - testInstance: Test instance to populate with results
    ///   - progressCallback: Callback for progress updates
    /// - Returns: Benchmark result with success status and timing data
    private func runOfficialBenchmark(
        engine: LLMInferenceEngineWrapper,
        instance: TestConfig,
        testInstance: TestInstance,
        progressCallback: @escaping (BenchmarkProgress) async -> Void
    ) async -> BenchmarkResult {
        
        return await withCheckedContinuation { continuation in
            var hasResumed = false
            
            engine.runOfficialBenchmark(
                withBackend: instance.backend,
                threads: instance.threads,
                useMmap: instance.useMmap,
                power: instance.power,
                precision: instance.precision,
                memory: instance.memory,
                dynamicOption: instance.dynamicOption,
                nPrompt: instance.nPrompt,
                nGenerate: instance.nGenerate,
                nRepeat: instance.nRepeat,
                kvCache: instance.kvCache,
                progressCallback: { [self] progressInfo in
                    // Convert Objective-C BenchmarkProgressInfo to Swift BenchmarkProgress
                    let swiftProgress = BenchmarkProgress(
                        progress: Int(progressInfo.progress),
                        statusMessage: progressInfo.statusMessage,
                        progressType: convertProgressType(progressInfo.progressType),
                        currentIteration: Int(progressInfo.currentIteration),
                        totalIterations: Int(progressInfo.totalIterations),
                        nPrompt: Int(progressInfo.nPrompt),
                        nGenerate: Int(progressInfo.nGenerate),
                        runTimeSeconds: progressInfo.runTimeSeconds,
                        prefillTimeSeconds: progressInfo.prefillTimeSeconds,
                        decodeTimeSeconds: progressInfo.decodeTimeSeconds,
                        prefillSpeed: progressInfo.prefillSpeed,
                        decodeSpeed: progressInfo.decodeSpeed
                    )
                    
                    Task {
                        await progressCallback(swiftProgress)
                    }
                },
                errorCallback: { errorMessage in
                    if !hasResumed {
                        hasResumed = true
                        let result = BenchmarkResult(
                            testInstance: testInstance,
                            success: false,
                            errorMessage: errorMessage
                        )
                        continuation.resume(returning: result)
                    }
                },
                iterationCompleteCallback: { detailedStats in
                    // Log detailed stats if needed
                    print("Benchmark iteration complete: \(detailedStats)")
                },
                completeCallback: { benchmarkResult in
                    if !hasResumed {
                        hasResumed = true
                        
                        // Update test instance with timing results
                        testInstance.prefillUs = benchmarkResult.prefillTimesUs.map { $0.int64Value }
                        testInstance.decodeUs = benchmarkResult.decodeTimesUs.map { $0.int64Value }
                        testInstance.samplesUs = benchmarkResult.sampleTimesUs.map { $0.int64Value }
                        
                        let result = BenchmarkResult(
                            testInstance: testInstance,
                            success: benchmarkResult.success,
                            errorMessage: benchmarkResult.errorMessage
                        )
                        continuation.resume(returning: result)
                    }
                }
            )
        }
    }
    
    // MARK: - Helper Methods & Configuration
    
    /// Converts Objective-C progress type to Swift enum
    /// - Parameter objcType: Objective-C progress type
    /// - Returns: Corresponding Swift ProgressType
    private func convertProgressType(_ objcType: BenchmarkProgressType) -> ProgressType {
        switch objcType {
        case .unknown:
            return .unknown
        case .initializing:
            return .initializing
        case .warmingUp:
            return .warmingUp
        case .runningTest:
            return .runningTest
        case .processingResults:
            return .processingResults
        case .completed:
            return .completed
        case .stopping:
            return .stopping
        @unknown default:
            return .unknown
        }
    }
    
    /// Generates test instances by combining runtime and test parameters
    /// - Parameters:
    ///   - runtimeParams: Runtime configuration parameters
    ///   - testParams: Test scenario parameters
    /// - Returns: Array of test configurations for execution
    private func generateTestInstances(
        runtimeParams: RuntimeParameters,
        testParams: TestParameters
    ) -> [TestConfig] {
        var instances: [TestConfig] = []
        
        for backend in runtimeParams.backends {
            for thread in runtimeParams.threads {
                for power in runtimeParams.power {
                    for precision in runtimeParams.precision {
                        for memory in runtimeParams.memory {
                            for dynamicOption in runtimeParams.dynamicOption {
                                for repeatCount in testParams.nRepeat {
                                    for (nPrompt, nGenerate) in testParams.nPrompGen {
                                        instances.append(TestConfig(
                                            configPath: "", // Will be set based on model
                                            backend: backend,
                                            threads: thread,
                                            useMmap: runtimeParams.useMmap,
                                            power: power,
                                            precision: precision,
                                            memory: memory,
                                            dynamicOption: dynamicOption,
                                            nPrompt: nPrompt,
                                            nGenerate: nGenerate,
                                            nRepeat: repeatCount,
                                            kvCache: testParams.kvCache == "true"
                                        ))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return instances
    }
}

// MARK: - Test Configuration

/**
 * Structure containing configuration parameters for a single benchmark test.
 * Combines runtime settings and test parameters into a complete test specification.
 */
struct TestConfig {
    let configPath: String
    let backend: Int
    let threads: Int
    let useMmap: Bool
    let power: Int
    let precision: Int
    let memory: Int
    let dynamicOption: Int
    let nPrompt: Int
    let nGenerate: Int
    let nRepeat: Int
    let kvCache: Bool
}
