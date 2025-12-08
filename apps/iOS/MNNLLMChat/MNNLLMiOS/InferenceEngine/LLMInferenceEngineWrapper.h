//
//  LLMInferenceEngineWrapper.h
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

#ifndef LLMInferenceEngineWrapper_h
#define LLMInferenceEngineWrapper_h

#import <Foundation/Foundation.h>
#if __has_include(<UIKit/UIKit.h>)
#import <UIKit/UIKit.h>
#elif __has_include(<AppKit/AppKit.h>)
#import <AppKit/AppKit.h>
#define UIImage NSImage
#endif

NS_ASSUME_NONNULL_BEGIN

typedef void (^CompletionHandler)(BOOL success);
typedef void (^OutputHandler)(NSString * _Nonnull output);

// MARK: - Benchmark Related Types

/**
 * Progress type enumeration for structured benchmark reporting
 */
typedef NS_ENUM(NSInteger, BenchmarkProgressType) {
    BenchmarkProgressTypeUnknown = 0,
    BenchmarkProgressTypeInitializing = 1,
    BenchmarkProgressTypeWarmingUp = 2,
    BenchmarkProgressTypeRunningTest = 3,
    BenchmarkProgressTypeProcessingResults = 4,
    BenchmarkProgressTypeCompleted = 5,
    BenchmarkProgressTypeStopping = 6
};

/**
 * Structured progress information for benchmark
 */
@interface BenchmarkProgressInfo : NSObject

@property (nonatomic, assign) NSInteger progress;              // 0-100
@property (nonatomic, strong) NSString *statusMessage;        // Status description
@property (nonatomic, assign) BenchmarkProgressType progressType;
@property (nonatomic, assign) NSInteger currentIteration;
@property (nonatomic, assign) NSInteger totalIterations;
@property (nonatomic, assign) NSInteger nPrompt;
@property (nonatomic, assign) NSInteger nGenerate;
@property (nonatomic, assign) float runTimeSeconds;
@property (nonatomic, assign) float prefillTimeSeconds;
@property (nonatomic, assign) float decodeTimeSeconds;
@property (nonatomic, assign) float prefillSpeed;
@property (nonatomic, assign) float decodeSpeed;

@end

/**
 * Benchmark result structure
 */
@interface BenchmarkResult : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, strong, nullable) NSString *errorMessage;
@property (nonatomic, strong) NSArray<NSNumber *> *prefillTimesUs;
@property (nonatomic, strong) NSArray<NSNumber *> *decodeTimesUs;
@property (nonatomic, strong) NSArray<NSNumber *> *sampleTimesUs;
@property (nonatomic, assign) NSInteger promptTokens;
@property (nonatomic, assign) NSInteger generateTokens;
@property (nonatomic, assign) NSInteger repeatCount;
@property (nonatomic, assign) BOOL kvCacheEnabled;

@end

// Benchmark callback blocks
typedef void (^BenchmarkProgressCallback)(BenchmarkProgressInfo *progressInfo);
typedef void (^BenchmarkErrorCallback)(NSString *error);
typedef void (^BenchmarkIterationCompleteCallback)(NSString *detailedStats);
typedef void (^BenchmarkCompleteCallback)(BenchmarkResult *result);

/**
 * LLMInferenceEngineWrapper - A high-level Objective-C wrapper for MNN LLM inference engine
 * 
 * This class provides a convenient interface for integrating MNN's Large Language Model
 * inference capabilities into iOS applications with enhanced error handling, performance
 * optimization, and thread safety.
 */
@interface LLMInferenceEngineWrapper : NSObject

/**
 * Initialize the LLM inference engine with a model path
 * 
 * @param modelPath The file system path to the model directory
 * @param completion Completion handler called with success/failure status
 * @return Initialized instance of LLMInferenceEngineWrapper
 */
- (instancetype)initWithModelPath:(NSString *)modelPath completion:(CompletionHandler)completion;

/**
 * Process user input and generate streaming LLM response
 * 
 * @param input The user's input text to process
 * @param output Callback block that receives streaming output chunks
 */
- (void)processInput:(NSString *)input withOutput:(OutputHandler)output;

/**
 * Process user input and generate streaming LLM response with optional performance output
 * 
 * @param input The user's input text to process
 * @param output Callback block that receives streaming output chunks
 * @param showPerformance Whether to output performance statistics after response completion
 */
- (void)processInput:(NSString *)input withOutput:(OutputHandler)output showPerformance:(BOOL)showPerformance;

/**
 * Process multimodal input (text + images) using MNN's MultimodalPrompt API.
 *
 * @param promptTemplate Template string containing <img>placeholder</img> tags.
 * @param images Dictionary mapping placeholder keys to UIImage objects.
 * @param output Callback block receiving streaming output chunks.
 * @param showPerformance Whether to show performance stats upon completion.
 */
- (void)processMultimodalInput:(NSString *)promptTemplate
                        images:(NSDictionary<NSString *, UIImage *> *)images
                    withOutput:(OutputHandler)output
               showPerformance:(BOOL)showPerformance;

/// Update maximum frames extracted for each video.
- (void)setVideoMaxFrames:(NSInteger)frames;

/// Set audio output enabled/disabled (for Omni models)
- (void)setEnableAudioOutput:(BOOL)enable;

/// Set talker speaker (for Omni models)
- (void)setTalkerSpeaker:(NSString *)speaker;

/// Set audio waveform callback for receiving PCM float data
- (void)setAudioWaveformCallback:(BOOL (^)(const float *data, size_t size, BOOL isLastChunk))callback;

/**
 * Add chat prompts from an array of dictionaries to the conversation history
 * 
 * @param array NSArray containing NSDictionary objects with chat messages
 */
- (void)addPromptsFromArray:(NSArray<NSDictionary *> *)array;

/**
 * Set the configuration for the LLM engine using a JSON string
 * 
 * @param jsonStr JSON string containing configuration parameters
 */
- (void)setConfigWithJSONString:(NSString *)jsonStr;

/**
 * Set thinking mode for the LLM engine
 * 
 * @param enabled Whether to enable thinking mode
 */
- (void)setThinkingModeEnabled:(BOOL)enabled;

/**
 * Check if model is ready for inference
 * 
 * @return YES if model is loaded and ready
 */
- (BOOL)isModelReady;

/**
 * Get current processing status
 * 
 * @return YES if currently processing an inference request
 */
- (BOOL)isProcessing;

/**
 * Cancel ongoing inference (if supported)
 */
- (void)cancelInference;

/**
 * Get chat history count
 * 
 * @return Number of messages in chat history
 */
- (NSUInteger)getChatHistoryCount;

/**
 * Clear chat history
 */
- (void)clearChatHistory;

// MARK: - Benchmark Methods

/**
 * Run official benchmark following llm_bench.cpp approach
 * 
 * @param backend Backend type (0 for CPU)
 * @param threads Number of threads
 * @param useMmap Whether to use memory mapping
 * @param power Power setting
 * @param precision Precision setting (2 for low precision)
 * @param memory Memory setting (2 for low memory)
 * @param dynamicOption Dynamic optimization option
 * @param nPrompt Number of prompt tokens
 * @param nGenerate Number of tokens to generate
 * @param nRepeat Number of repetitions
 * @param kvCache Whether to use KV cache
 * @param progressCallback Progress update callback
 * @param errorCallback Error callback
 * @param iterationCompleteCallback Iteration completion callback
 * @param completeCallback Final completion callback
 */
- (void)runOfficialBenchmarkWithBackend:(NSInteger)backend
                                threads:(NSInteger)threads
                                useMmap:(BOOL)useMmap
                                  power:(NSInteger)power
                              precision:(NSInteger)precision
                                 memory:(NSInteger)memory
                          dynamicOption:(NSInteger)dynamicOption
                                nPrompt:(NSInteger)nPrompt
                              nGenerate:(NSInteger)nGenerate
                                nRepeat:(NSInteger)nRepeat
                                kvCache:(BOOL)kvCache
                       progressCallback:(BenchmarkProgressCallback _Nullable)progressCallback
                          errorCallback:(BenchmarkErrorCallback _Nullable)errorCallback
               iterationCompleteCallback:(BenchmarkIterationCompleteCallback _Nullable)iterationCompleteCallback
                       completeCallback:(BenchmarkCompleteCallback _Nullable)completeCallback;

/**
 * Stop running benchmark
 */
- (void)stopBenchmark;

/**
 * Check if benchmark is currently running
 * 
 * @return YES if benchmark is running
 */
- (BOOL)isBenchmarkRunning;

/**
 * Process multiple prompts in a single batch and return their responses.
 *
 * This method runs each prompt independently, clears the chat history per prompt,
 * and collects the generated outputs without streaming UI callbacks.
 *
 * @param prompts An array of input prompt strings to process
 * @param completion Completion block called on the main thread with an array of
 *                   response strings in the same order as the input prompts
 */
- (void)processBatchPrompts:(NSArray<NSString *> *)prompts
                 completion:(void (^)(NSArray<NSString *> *responses))completion;

@end

NS_ASSUME_NONNULL_END

#endif /* LLMInferenceEngineWrapper_h */
