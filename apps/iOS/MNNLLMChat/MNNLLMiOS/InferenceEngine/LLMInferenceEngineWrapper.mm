//
//  LLMInferenceEngineWrapper.m
//  mnn-llm
//  Modified by 游薪渝(揽清) on 2025/7/7.
//  Created by wangzhaode on 2023/12/14.
//

/**
 * LLMInferenceEngineWrapper - A high-level Objective-C wrapper for MNN LLM inference engine
 *
 * This class provides a convenient interface for integrating MNN's Large Language Model
 * inference capabilities into iOS applications. It handles model loading, configuration,
 * text processing, and streaming output with proper memory management and error handling.
 *
 * Key Features:
 * - Asynchronous model loading with completion callbacks
 * - Streaming text generation with real-time output
 * - Configurable inference parameters through JSON
 * - Memory-mapped model loading for efficiency
 * - Chat history management and conversation context
 * - Benchmarking capabilities for performance testing
 *
 * Usage Examples:
 *
 * 1. Basic Model Loading and Inference:
 * ```objc
 * LLMInferenceEngineWrapper *engine = [[LLMInferenceEngineWrapper alloc]
 *     initWithModelPath:@"/path/to/model"
 *     completion:^(BOOL success) {
 *         if (success) {
 *             NSLog(@"Model loaded successfully");
 *         }
 *     }];
 *
 * [engine processInput:@"Hello, how are you?"
 *           withOutput:^(NSString *output) {
 *               NSLog(@"AI Response: %@", output);
 *           }];
 * ```
 *
 * 2. Configuration with Custom Parameters:
 * ```objc
 * NSString *config = @"{\"temperature\":0.7,\"max_tokens\":100}";
 * [engine setConfigWithJSONString:config];
 * ```
 *
 * 3. Chat History Management:
 * ```objc
 * NSArray *chatHistory = @[
 *     @{@"user": @"What is AI?"},
 *     @{@"assistant": @"AI stands for Artificial Intelligence..."}
 * ];
 * [engine addPromptsFromArray:chatHistory];
 * ```
 *
 * Architecture:
 * - Built on top of MNN's C++ LLM inference engine
 * - Uses smart pointers for automatic memory management
 * - Implements custom stream buffer for real-time text output
 * - Supports both bundled and external model loading
 */

#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include "MNN/expr/ExecutorScope.hpp"

#import <Foundation/Foundation.h>
#import "LLMInferenceEngineWrapper.h"

// Conditional include for MNN headers
#ifdef __has_include
  #if __has_include(<MNN/llm/llm.hpp>)
    #include <MNN/llm/llm.hpp>
    using namespace MNN::Transformer;
  #else
    // Fallback declarations when MNN headers are not available
    namespace MNN {
        namespace Transformer {
            class Llm {
             public:
                 static Llm* createLLM(const std::string& config_path);
                 virtual void set_config(const std::string& config) = 0;
                 virtual void load() = 0;
                 virtual void response(const std::string& input_str, std::ostream* os = nullptr, const char* end_with = nullptr) = 0;
                 virtual void response(const std::vector<std::pair<std::string, std::string>>& history, std::ostream* os = nullptr, const char* end_with = nullptr, int max_new_tokens = 999999) = 0;
                 virtual void response(const std::vector<int>& tokens, std::ostream* os = nullptr, const char* end_with = nullptr, int max_new_tokens = 999999) = 0;
                 virtual void reset() = 0;
                 virtual bool stopped() = 0;
                 virtual int generate(int max_token_number = 0) = 0;
                 struct LlmContext {
                     int prompt_len;
                     int gen_seq_len;
                     int64_t prefill_us;
                     int64_t decode_us;
                 };
                 virtual LlmContext* getContext() = 0;
                 virtual ~Llm() = default;
             };
        }
    }
    using namespace MNN::Transformer;
  #endif
#else
  // Fallback for older compilers
  namespace MNN {
      namespace Transformer {
          class Llm {
           public:
               static Llm* createLLM(const std::string& config_path);
               virtual void set_config(const std::string& config) = 0;
               virtual void load() = 0;
               virtual void response(const std::string& input_str, std::ostream* os = nullptr, const char* end_with = nullptr) = 0;
               virtual void response(const std::vector<std::pair<std::string, std::string>>& history, std::ostream* os = nullptr, const char* end_with = nullptr, int max_new_tokens = 512) = 0;
               virtual void response(const std::vector<int>& tokens, std::ostream* os = nullptr, const char* end_with = nullptr, int max_new_tokens = 512) = 0;
               virtual void reset() = 0;
               virtual bool stopped() = 0;
               virtual int generate(int max_token_number = 0) = 0;
               struct LlmContext {
                   int prompt_len;
                   int gen_seq_len;
                   int64_t prefill_us;
                   int64_t decode_us;
               };
               virtual LlmContext* getContext() = 0;
               virtual ~Llm() = default;
           };
      }
  }
  using namespace MNN::Transformer;
#endif

using ChatMessage = std::pair<std::string, std::string>;

// MARK: - Benchmark Progress Info Implementation

@implementation BenchmarkProgressInfo

- (instancetype)init {
    self = [super init];
    if (self) {
        _progress = 0;
        _statusMessage = @"";
        _progressType = BenchmarkProgressTypeUnknown;
        _currentIteration = 0;
        _totalIterations = 0;
        _nPrompt = 0;
        _nGenerate = 0;
        _runTimeSeconds = 0.0f;
        _prefillTimeSeconds = 0.0f;
        _decodeTimeSeconds = 0.0f;
        _prefillSpeed = 0.0f;
        _decodeSpeed = 0.0f;
    }
    return self;
}

@end

// MARK: - Benchmark Result Implementation

@implementation BenchmarkResult

- (instancetype)init {
    self = [super init];
    if (self) {
        _success = NO;
        _errorMessage = nil;
        _prefillTimesUs = @[];
        _decodeTimesUs = @[];
        _sampleTimesUs = @[];
        _promptTokens = 0;
        _generateTokens = 0;
        _repeatCount = 0;
        _kvCacheEnabled = NO;
    }
    return self;
}

@end


/**
 * C++ Benchmark result structure following Android implementation
 */
struct BenchmarkResultCpp {
    bool success;
    std::string error_message;
    std::vector<int64_t> prefill_times_us;
    std::vector<int64_t> decode_times_us;
    std::vector<int64_t> sample_times_us;
    int prompt_tokens;
    int generate_tokens;
    int repeat_count;
    bool kv_cache_enabled;
};

/**
 * C++ Benchmark progress info structure following Android implementation
 */
struct BenchmarkProgressInfoCpp {
    int progress;
    std::string statusMessage;
    int progressType;
    int currentIteration;
    int totalIterations;
    int nPrompt;
    int nGenerate;
    float runTimeSeconds;
    float prefillTimeSeconds;
    float decodeTimeSeconds;
    float prefillSpeed;
    float decodeSpeed;
    
    BenchmarkProgressInfoCpp() : progress(0), statusMessage(""), progressType(0),
                                currentIteration(0), totalIterations(0), nPrompt(0), nGenerate(0),
                                runTimeSeconds(0.0f), prefillTimeSeconds(0.0f), decodeTimeSeconds(0.0f),
                                prefillSpeed(0.0f), decodeSpeed(0.0f) {}
};

// MARK: - C++ Benchmark Implementation

/**
 * C++ Benchmark callback structure following Android implementation
 */
struct BenchmarkCallback {
    std::function<void(const BenchmarkProgressInfoCpp& progressInfo)> onProgress;
    std::function<void(const std::string& error)> onError;
    std::function<void(const std::string& detailed_stats)> onIterationComplete;
};


/**
 * Enhanced LlmStreamBuffer with improved performance and error handling
 */
class OptimizedLlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;
    
    OptimizedLlmStreamBuffer(CallBack callback) : callback_(callback) {
        buffer_.reserve(1024); // Pre-allocate buffer for better performance
    }
    
    ~OptimizedLlmStreamBuffer() {
        flushBuffer();
    }

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (!callback_ || n <= 0) {
            return n;
        }
        
        try {
            buffer_.append(s, n);
            
            const size_t BUFFER_THRESHOLD = 64;
            bool shouldFlush = buffer_.size() >= BUFFER_THRESHOLD;
            
            if (!shouldFlush && n > 0) {
                shouldFlush = checkForFlushTriggers(s, n);
            }
            
            if (shouldFlush) {
                flushBuffer();
            }
            
            return n;
        }
        catch (const std::exception& e) {
            NSLog(@"Error in stream buffer: %s", e.what());
            return -1;
        }
    }

private:
    void flushBuffer() {
        if (callback_ && !buffer_.empty()) {
            callback_(buffer_.c_str(), buffer_.size());
            buffer_.clear();
        }
    }
    
    bool checkForFlushTriggers(const char* s, std::streamsize n) {
        // Check ASCII punctuation
        char lastChar = s[n-1];
        if (lastChar == '\n' ||
            lastChar == '\r' ||
            lastChar == '\t' ||
            lastChar == '.' ||
            lastChar == ',' ||
            lastChar == ';' ||
            lastChar == ':' ||
            lastChar == '!' ||
            lastChar == '?') {
            return true;
        }
        
        // Check Unicode punctuation
        return checkUnicodePunctuation();
    }
    
    bool checkUnicodePunctuation() {
        if (buffer_.size() >= 3) {
            const char* bufferEnd = buffer_.c_str() + buffer_.size() - 3;
            
            // Chinese punctuation marks (3-byte UTF-8)
            static const std::vector<std::string> chinesePunctuation = {
                "\xE3\x80\x82",     // 。
                "\xEF\xBC\x8C",     // ，
                "\xEF\xBC\x9B",     // ；
                "\xEF\xBC\x9A",     // ：
                "\xEF\xBC\x81",     // ！
                "\xEF\xBC\x9F",     // ？
                "\xE2\x80\xA6",     // …
            };
            
            for (const auto& punct : chinesePunctuation) {
                if (memcmp(bufferEnd, punct.c_str(), 3) == 0) {
                    return true;
                }
            }
        }
        
        // Check 2-byte punctuation
        if (buffer_.size() >= 2) {
            const char* bufferEnd = buffer_.c_str() + buffer_.size() - 2;
            if (memcmp(bufferEnd, "\xE2\x80\x93", 2) == 0 ||  // –
                memcmp(bufferEnd, "\xE2\x80\x94", 2) == 0) {  // —
                return true;
            }
        }
        
        return false;
    }
    
    CallBack callback_ = nullptr;
    std::string buffer_; // Buffer for accumulating output
};

@implementation LLMInferenceEngineWrapper {
    std::shared_ptr<MNN::Transformer::Llm> _llm;
    std::vector<ChatMessage> _history;
    std::mutex _historyMutex;
    std::atomic<bool> _isProcessing;
    std::atomic<bool> _isBenchmarkRunning;
    std::atomic<bool> _shouldStopBenchmark;
    std::atomic<bool> _shouldStopInference;
    NSString *_modelPath;
}

/**
 * Initializes the LLM inference engine with a model path
 *
 * This method asynchronously loads the LLM model from the specified path
 * and calls the completion handler on the main queue when finished.
 *
 * @param modelPath The file system path to the model directory
 * @param completion Completion handler called with success/failure status
 * @return Initialized instance of LLMInferenceEngineWrapper
 */
- (instancetype)initWithModelPath:(NSString *)modelPath completion:(CompletionHandler)completion {
    self = [super init];
    if (self) {
        _modelPath = [modelPath copy];
        _isProcessing = false;
        _isBenchmarkRunning = false;
        _shouldStopBenchmark = false;
        _shouldStopInference = false;
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
            BOOL success = [self loadModelFromPath:modelPath];
            
            dispatch_async(dispatch_get_main_queue(), ^{
                if (completion) {
                    completion(success);
                }
            });
        });
    }
    return self;
}

/**
 * Utility function to remove a directory and all its contents
 *
 * @param path The directory path to remove
 * @return true if successful, false otherwise
 */
bool remove_directory_safely(const std::string& path) {
    @try {
        NSString *pathStr = [NSString stringWithUTF8String:path.c_str()];
        NSFileManager *fileManager = [NSFileManager defaultManager];
        
        if ([fileManager fileExistsAtPath:pathStr]) {
            NSError *error = nil;
            BOOL success = [fileManager removeItemAtPath:pathStr error:&error];
            if (!success && error) {
                NSLog(@"Error removing directory %s: %@", path.c_str(), error.localizedDescription);
                return false;
            }
            return success;
        }
        return true;
    } @catch (NSException *exception) {
        NSLog(@"Exception removing directory %s: %@", path.c_str(), exception.reason);
        return false;
    }
}

/**
 * Validates model path and configuration
 *
 * @param modelPath The path to validate
 * @return YES if path is valid and contains required files
 */
- (BOOL)validateModelPath:(NSString *)modelPath {
    if (!modelPath || modelPath.length == 0) {
        NSLog(@"Error: Model path is nil or empty");
        return NO;
    }
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    BOOL isDirectory;
    
    if (![fileManager fileExistsAtPath:modelPath isDirectory:&isDirectory] || !isDirectory) {
        NSLog(@"Error: Model path does not exist or is not a directory: %@", modelPath);
        return NO;
    }
    
    NSString *configPath = [modelPath stringByAppendingPathComponent:@"config.json"];
    if (![fileManager fileExistsAtPath:configPath]) {
        NSLog(@"Error: config.json not found at path: %@", configPath);
        return NO;
    }
    
    return YES;
}

/**
 * Loads the LLM model from the application bundle
 *
 * This method is used for testing with models bundled within the app.
 * It sets up the model with default configuration and temporary directory.
 *
 * @return YES if model loading succeeds, NO otherwise
 */
- (BOOL)loadModel {
    @try {
        if (_llm) {
            NSLog(@"Warning: Model already loaded");
            return YES;
        }
        
        NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
        std::string model_dir = [bundleDirectory UTF8String];
        std::string config_path = model_dir + "/config.json";
        
        _llm.reset(MNN::Transformer::Llm::createLLM(config_path));
        if (!_llm) {
            NSLog(@"Error: Failed to create LLM from bundle");
            return NO;
        }
        
        NSString *tempDirectory = NSTemporaryDirectory();
        std::string configStr = "{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\", \"use_mmap\":true}";
        _llm->set_config(configStr);
        _llm->load();
        
        NSLog(@"Model loaded successfully from bundle");
        return YES;
    }
    @catch (NSException *exception) {
        NSLog(@"Exception during model loading: %@", exception.reason);
        return NO;
    }
}

/**
 * Loads the LLM model from a specified file system path
 *
 * This method handles the complete model loading process including:
 * - Path validation and error checking
 * - Reading model configuration from config.json
 * - Setting up temporary directories for model operations
 * - Configuring memory mapping settings
 * - Loading the model into memory with proper error handling
 *
 * @param modelPath The file system path to the model directory
 * @return YES if model loading succeeds, NO otherwise
 */
- (BOOL)loadModelFromPath:(NSString *)modelPath {
    @try {
        if (_llm) {
            NSLog(@"Warning: Model already loaded");
            return YES;
        }
        
        if (![self validateModelPath:modelPath]) {
            return NO;
        }
        
        std::string config_path = std::string([modelPath UTF8String]) + "/config.json";
        
        // Read and parse configuration with error handling
        NSError *error = nil;
        NSData *configData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:config_path.c_str()]];
        if (!configData) {
            NSLog(@"Error: Failed to read config file at %s", config_path.c_str());
            return NO;
        }
        
        NSDictionary *configDict = [NSJSONSerialization JSONObjectWithData:configData options:0 error:&error];
        if (error) {
            NSLog(@"Error parsing config JSON: %@", error.localizedDescription);
            return NO;
        }
        
        MNN::BackendConfig backendConfig;
        auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
        MNN::Express::ExecutorScope s(executor);
        
        // Get memory mapping setting with default fallback
        BOOL useMmap = configDict[@"use_mmap"] == nil ? YES : [configDict[@"use_mmap"] boolValue];
        
        // Create LLM instance with error checking
        _llm.reset(MNN::Transformer::Llm::createLLM(config_path));
        if (!_llm) {
            NSLog(@"Error: Failed to create LLM instance from config: %s", config_path.c_str());
            return NO;
        }
        
        // Setup temporary directory with improved error handling
        // Use iOS system temporary directory instead of model path (which is read-only in Bundle)
        NSString *tempDir = NSTemporaryDirectory();
        NSString *modelName = [[modelPath lastPathComponent] stringByDeletingPathExtension];
        NSString *tempDirPath = [tempDir stringByAppendingPathComponent:[NSString stringWithFormat:@"MNN_%@_temp", modelName]];
        std::string temp_directory_path = [tempDirPath UTF8String];
        
        // Clean up existing temp directory
        if (!remove_directory_safely(temp_directory_path)) {
            NSLog(@"Warning: Failed to remove existing temp directory, continuing...");
        }
        
        // Create new temp directory in system temp location
        if (mkdir(temp_directory_path.c_str(), 0755) != 0 && errno != EEXIST) {
            NSLog(@"Error: Failed to create temp directory: %s, errno: %d", temp_directory_path.c_str(), errno);
            return NO;
        }
        
        NSLog(@"Created temp directory at: %s", temp_directory_path.c_str());
        
        // Configure LLM with proper error handling
        bool useMmapCpp = (useMmap == YES);
        std::string configStr = "{\"tmp_path\":\"" + temp_directory_path + "\", \"use_mmap\":" + (useMmapCpp ? "true" : "false") + "}";
        
        _llm->set_config(configStr);
        _llm->load();
        
        NSLog(@"Model loaded successfully from path: %@", modelPath);
        return YES;
    }
    @catch (NSException *exception) {
        NSLog(@"Exception during model loading: %@", exception.reason);
        _llm.reset();
        return NO;
    }
}

/**
 * Sets the configuration for the LLM engine using a JSON string
 *
 * This method allows runtime configuration of various LLM parameters
 * such as temperature, max tokens, sampling methods, etc.
 *
 * @param jsonStr JSON string containing configuration parameters
 */
- (void)setConfigWithJSONString:(NSString *)jsonStr {
    if (!_llm) {
        NSLog(@"Error: LLM not initialized, cannot set configuration");
        return;
    }
    
    if (!jsonStr || jsonStr.length == 0) {
        NSLog(@"Error: JSON string is nil or empty");
        return;
    }
    
    @try {
        // Validate JSON format
        NSError *error = nil;
        NSData *jsonData = [jsonStr dataUsingEncoding:NSUTF8StringEncoding];
        [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
        
        if (error) {
            NSLog(@"Error: Invalid JSON configuration: %@", error.localizedDescription);
            return;
        }
        
        const char *cString = [jsonStr UTF8String];
        std::string stdString(cString);
        _llm->set_config(stdString);
        
        NSLog(@"Configuration updated successfully");
    }
    @catch (NSException *exception) {
        NSLog(@"Exception while setting configuration: %@", exception.reason);
    }
}

/**
 * Set thinking mode for the LLM engine
 * 
 * @param enabled Whether to enable thinking mode
 */
- (void)setThinkingModeEnabled:(BOOL)enabled {
    if (!_llm) {
        NSLog(@"Warning: LLM engine not initialized, cannot set thinking mode");
        return;
    }
    
    try {
        std::string configJson = R"({
            "jinja": {
                "context": {
                    "enable_thinking":)" + std::string(enabled ? "true" : "false") + R"(
                }
            }
        })";
        
        _llm->set_config(configJson);
        
        NSLog(@"Thinking mode %@", enabled ? @"enabled" : @"disabled");
        
    } catch (const std::exception& e) {
        NSLog(@"Error setting thinking mode: %s", e.what());
    } catch (...) {
        NSLog(@"Unknown error occurred while setting thinking mode");
    }
}

/**
 * Processes user input and generates streaming LLM response with enhanced error handling
 *
 * This method handles the main inference process by:
 * - Validating input parameters and model state
 * - Setting up streaming output callback with error handling
 * - Adding user input to chat history thread-safely
 * - Executing LLM inference with streaming output
 * - Handling special commands like benchmarking
 *
 * @param input The user's input text to process
 * @param output Callback block that receives streaming output chunks
 */
- (void)processInput:(NSString *)input withOutput:(OutputHandler)output {
    [self processInput:input withOutput:output showPerformance:NO];
}

/**
 * Processes user input and generates streaming LLM response with optional performance output
 *
 * @param input The user's input text to process
 * @param output Callback block that receives streaming output chunks
 * @param showPerformance Whether to output performance statistics after response completion
 */
- (void)processInput:(NSString *)input withOutput:(OutputHandler)output showPerformance:(BOOL)showPerformance {
    if (!_llm) {
        if (output) {
            output(@"Error: Model not loaded. Please initialize the model first.");
        }
        return;
    }
    
    if (!input || input.length == 0) {
        if (output) {
            output(@"Error: Input text is empty.");
        }
        return;
    }
    
    if (_isProcessing.load()) {
        if (output) {
            output(@"Error: Another inference is already in progress.");
        }
        return;
    }
    
    // Performance measurement initialization
    auto inference_start_time = std::chrono::high_resolution_clock::now();
    
    // Get initial context state BEFORE inference starts
    auto* context = _llm->getContext();
    int initial_prompt_len = 0;
    int initial_decode_len = 0;
    int64_t initial_prefill_time = 0;
    int64_t initial_decode_time = 0;
    
    if (context && showPerformance) {
        initial_prompt_len = context->prompt_len;
        initial_decode_len = context->gen_seq_len;
        initial_prefill_time = context->prefill_us;
        initial_decode_time = context->decode_us;
    }

    _isProcessing = true;
    
    // Store reference for block execution
    LLMInferenceEngineWrapper *blockSelf = self;
    
    // Use high priority queue for better responsiveness
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        // Check if object is still valid before proceeding
        if (!blockSelf || !blockSelf->_llm) {
            NSLog(@"LLMInferenceEngineWrapper was deallocated or model unloaded during inference");
            return;
        }
        
        @try {
            auto inference_start_time = std::chrono::high_resolution_clock::now();
            
            OptimizedLlmStreamBuffer::CallBack callback = [output](const char* str, size_t len) {
                if (output && str && len > 0) {
                    @autoreleasepool {
                        NSString *nsOutput = [[NSString alloc] initWithBytes:str
                                                                        length:len
                                                                      encoding:NSUTF8StringEncoding];
                        if (nsOutput) {
                            dispatch_async(dispatch_get_main_queue(), ^{
                                output(nsOutput);
                            });
                        }
                    }
                }
            };
            
            OptimizedLlmStreamBuffer streambuf(callback);
            std::ostream os(&streambuf);
            
            // Thread-safe history management
            {
                std::lock_guard<std::mutex> lock(blockSelf->_historyMutex);
                blockSelf->_history.emplace_back(ChatMessage("user", [input UTF8String]));
            }
            
            std::string inputStr = [input UTF8String];
            #ifdef DEBUG
            if (inputStr == "benchmark") {
                [blockSelf performBenchmarkWithOutput:&os];
            } else {
            #else
            {
            #endif
                // Get initial context state for performance measurement
                auto context = blockSelf->_llm->getContext();
                int initial_prompt_len = context->prompt_len;
                int initial_decode_len = context->gen_seq_len;
                int64_t initial_prefill_time = context->prefill_us;
                int64_t initial_decode_time = context->decode_us;
                
                // Reset stop flag before starting inference
                blockSelf->_shouldStopInference = false;
                
                // Execute inference with enhanced stopped status checking
                @try {
                    // Debug information for prompt
                    std::string prompt_debug = "";
                    for (const auto& msg : blockSelf->_history) {
                        prompt_debug += msg.first + ": " + msg.second + "\n";
                    }
                    NSLog(@"submitNative prompt_string_for_debug:\n%s\nmax_new_tokens_: %d", prompt_debug.c_str(), 999999);
                    
                    // Start inference with initial response processing
                    blockSelf->_llm->response(blockSelf->_history, &os, "<eop>", 1);
                    int current_size = 1;
                    int max_new_tokens = 999999;
                    
                    // Continue generation with precise token-by-token control
                    while (!blockSelf->_shouldStopInference.load() && 
                           !blockSelf->_llm->stoped() &&
                           current_size < max_new_tokens) {
                        
                        // Generate single token for maximum control
                        blockSelf->_llm->generate(1);
                        current_size++;
                        
                        // Small delay to allow UI updates and stop signal processing
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    
                    // Send appropriate end signal based on stop reason
                    if (output) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            if (blockSelf->_shouldStopInference.load()) {
                                output(@"<stopped>");
                            } else {
                                output(@"<eop>");
                            }
                        });
                    }
                    
                    NSLog(@"Inference completed. Generated tokens: %d, Stopped by user: %s, Model stopped: %s", 
                          current_size, 
                          blockSelf->_shouldStopInference.load() ? "YES" : "NO",
                          blockSelf->_llm->stoped() ? "YES" : "NO");
                    
                } @catch (NSException *exception) {
                    NSLog(@"Exception during response generation: %@", exception.reason);
                    
                    // Send end signal even on error to unlock UI
                    if (output) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            output(@"<eop>");
                        });
                    }
                }
                
                // Calculate performance metrics if requested
                if (showPerformance && context) {
                    auto inference_end_time = std::chrono::high_resolution_clock::now();
                    auto total_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        inference_end_time - inference_start_time
                    );
                    
                    // Get final context state AFTER inference completes
                    int final_prompt_len = context->prompt_len;
                    int final_decode_len = context->gen_seq_len;
                    int64_t final_prefill_time = context->prefill_us;
                    int64_t final_decode_time = context->decode_us;
                    
                    // Calculate differences for this inference session
                    int current_prompt_len = final_prompt_len - initial_prompt_len;
                    int current_decode_len = final_decode_len - initial_decode_len;
                    int64_t current_prefill_time = final_prefill_time - initial_prefill_time;
                    int64_t current_decode_time = final_decode_time - initial_decode_time;
                    
                    // Convert microseconds to seconds
                    float prefill_s = static_cast<float>(current_prefill_time) / 1e6f;
                    float decode_s = static_cast<float>(current_decode_time) / 1e6f;
                    
                    // Calculate speeds (tokens per second)
                    float prefill_speed = (prefill_s > 0.001f) ?
                        static_cast<float>(current_prompt_len) / prefill_s : 0.0f;
                    float decode_speed = (decode_s > 0.001f) ?
                        static_cast<float>(current_decode_len) / decode_s : 0.0f;
                    
                    // Format performance results with better formatting
                    std::ostringstream performance_output;
                    performance_output << "\n\n > Performance Metrics:\n"
                                      << "Total inference time: " << total_inference_time.count() << " ms\n"
                                      << " Prompt tokens: " << current_prompt_len << "\n"
                                      << "Generated tokens: " << current_decode_len << "\n"
                                      << "Prefill time: " << std::fixed << std::setprecision(3) << prefill_s << " s\n"
                                      << "Decode time: " << std::fixed << std::setprecision(3) << decode_s << " s\n"
                                      << "Prefill speed: " << std::fixed << std::setprecision(1) << prefill_speed << " tok/s\n"
                                      << "Decode speed: " << std::fixed << std::setprecision(1) << decode_speed << " tok/s\n";
                    
                    // Add efficiency metrics
                    if (current_prompt_len > 0 && current_decode_len > 0) {
                        float total_tokens = static_cast<float>(current_prompt_len + current_decode_len);
                        float total_time_s = static_cast<float>(total_inference_time.count()) / 1000.0f;
                        float overall_speed = total_time_s > 0.001f ? total_tokens / total_time_s : 0.0f;
                        
                        performance_output << "> Overall speed: " << std::fixed << std::setprecision(1)
                                          << overall_speed << " tok/s\n";
                    }
                    
                    // Output performance results on main queue
                    std::string perf_str = performance_output.str();
                    if (output) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            NSString *perfOutput = [NSString stringWithUTF8String:perf_str.c_str()];
                            if (perfOutput) {
                                output(perfOutput);
                            }
                        });
                    }
                }
            }
        }
        @catch (NSException *exception) {
            NSLog(@"Exception during inference: %@", exception.reason);
            if (output) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    output([NSString stringWithFormat:@"Error: Inference failed - %@", exception.reason]);
                });
            }
        }
        @finally {
            blockSelf->_isProcessing = false;
        }
    });
}

/**
 * Performs benchmark testing with enhanced error handling and reporting
 *
 * @param os Output stream for benchmark results
 */
- (void)performBenchmarkWithOutput:(std::ostream *)os {
    @try {
        std::string model_dir = [[[NSBundle mainBundle] bundlePath] UTF8String];
        std::string prompt_file = model_dir + "/bench.txt";
        
        std::ifstream prompt_fs(prompt_file);
        if (!prompt_fs.is_open()) {
            *os << "Error: Could not open benchmark file at " << prompt_file << std::endl;
            return;
        }
        
        std::vector<std::string> prompts;
        std::string prompt;
        
        while (std::getline(prompt_fs, prompt)) {
            if (prompt.empty() || prompt.substr(0, 1) == "#") {
                continue;
            }
            
            // Process escape sequences
            std::string::size_type pos = 0;
            while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
                prompt.replace(pos, 2, "\n");
                pos += 1;
            }
            prompts.push_back(prompt);
        }
        
        if (prompts.empty()) {
            *os << "Error: No valid prompts found in benchmark file" << std::endl;
            return;
        }
        
        // Performance metrics
        int prompt_len = 0;
        int decode_len = 0;
        int64_t prefill_time = 0;
        int64_t decode_time = 0;
        
        auto context = _llm->getContext();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& p : prompts) {
            _llm->response(p, os, "\n");
            prompt_len += context->prompt_len;
            decode_len += context->gen_seq_len;
            prefill_time += context->prefill_us;
            decode_time += context->decode_us;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        float prefill_s = prefill_time / 1e6;
        float decode_s = decode_time / 1e6;
        
        *os << "\n#################################\n"
            << "Benchmark Results:\n"
            << "Total prompts processed: " << prompts.size() << "\n"
            << "Total time: " << total_time.count() << " ms\n"
            << "Prompt tokens: " << prompt_len << "\n"
            << "Decode tokens: " << decode_len << "\n"
            << "Prefill time: " << std::fixed << std::setprecision(2) << prefill_s << " s\n"
            << "Decode time: " << std::fixed << std::setprecision(2) << decode_s << " s\n"
            << "Prefill speed: " << std::fixed << std::setprecision(2)
            << (prefill_s > 0 ? prompt_len / prefill_s : 0) << " tok/s\n"
            << "Decode speed: " << std::fixed << std::setprecision(2)
            << (decode_s > 0 ? decode_len / decode_s : 0) << " tok/s\n"
            << "#################################\n";
        *os << "<eop>";
    }
    @catch (NSException *exception) {
        *os << "Error during benchmark: " << [exception.reason UTF8String] << std::endl;
    }
}

/**
 * Enhanced deallocation with proper cleanup and timeout
 */
- (void)dealloc {
    NSLog(@"LLMInferenceEngineWrapper deallocating...");
    
    // Actively cancel all operations first
    [self cancelInference];
    
    // Wait for any ongoing processing to complete with timeout
    int timeout = 100; // 1 second timeout (100 * 10ms)
    while ((_isProcessing.load() || _isBenchmarkRunning.load()) && timeout > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout--;
    }
    
    if (timeout <= 0) {
        NSLog(@"Warning: Dealloc timeout, forcing cleanup");
        _isProcessing = false;
        _isBenchmarkRunning = false;
    }
    
    {
        std::lock_guard<std::mutex> lock(_historyMutex);
        _history.clear();
    }
    
    _llm.reset();
    NSLog(@"LLMInferenceEngineWrapper deallocation complete");
}


/**
 * Enhanced chat history initialization with thread safety
 *
 * @param chatHistory Vector of strings representing alternating user/assistant messages
 */
- (void)init:(const std::vector<std::string>&)chatHistory {
    std::lock_guard<std::mutex> lock(_historyMutex);
    _history.clear();
    _history.emplace_back("system", "You are a helpful assistant.");
    
    for (size_t i = 0; i < chatHistory.size(); ++i) {
        _history.emplace_back(i % 2 == 0 ? "user" : "assistant", chatHistory[i]);
    }
    NSLog(@"Chat history initialized with %zu messages", chatHistory.size());
}

/**
 * Enhanced method for adding chat prompts from array with validation
 *
 * @param array NSArray containing NSDictionary objects with chat messages
 */
- (void)addPromptsFromArray:(NSArray<NSDictionary *> *)array {
    if (!array || array.count == 0) {
        NSLog(@"Warning: Empty or nil chat history array provided");
        return;
    }
    
    std::lock_guard<std::mutex> lock(_historyMutex);
    _history.clear();
    
    for (NSDictionary *dict in array) {
        if ([dict isKindOfClass:[NSDictionary class]]) {
            [self addPromptsFromDictionary:dict];
        } else {
            NSLog(@"Warning: Invalid dictionary in chat history array");
        }
    }
    NSLog(@"Added prompts from array with %lu items", (unsigned long)array.count);
}

/**
 * Enhanced method for adding prompts from dictionary with validation
 *
 * @param dictionary NSDictionary containing role-message key-value pairs
 */
- (void)addPromptsFromDictionary:(NSDictionary *)dictionary {
    if (!dictionary || dictionary.count == 0) {
        return;
    }
    
    for (NSString *key in dictionary) {
        NSString *value = dictionary[key];
        
        if (![key isKindOfClass:[NSString class]] || ![value isKindOfClass:[NSString class]]) {
            NSLog(@"Warning: Invalid key-value pair in chat dictionary");
            continue;
        }
        
        std::string keyString = [key UTF8String];
        std::string valueString = [value UTF8String];
        _history.emplace_back(ChatMessage(keyString, valueString));
    }
}

/**
 * Check if model is ready for inference
 *
 * @return YES if model is loaded and ready
 */
- (BOOL)isModelReady {
    return _llm != nullptr && !_isProcessing.load();
}

/**
 * Get current processing status
 *
 * @return YES if currently processing an inference request
 */
- (BOOL)isProcessing {
    return _isProcessing.load();
}

/**
 * Cancel ongoing inference (if supported)
 */
- (void)cancelInference {
    NSLog(@"Cancelling inference...");
    
    // Set all stop flags to true
    _shouldStopInference = true;
    _shouldStopBenchmark = true;
    
    // Force set processing states to false for immediate cleanup
    _isProcessing = false;
    _isBenchmarkRunning = false;
    
    NSLog(@"Inference cancellation completed - all flags set");
}

/**
 * Get chat history count
 *
 * @return Number of messages in chat history
 */
- (NSUInteger)getChatHistoryCount {
    std::lock_guard<std::mutex> lock(_historyMutex);
    return _history.size();
}

/**
 * Clear chat history
 */
- (void)clearChatHistory {
    std::lock_guard<std::mutex> lock(_historyMutex);
    _history.clear();
    NSLog(@"Chat history cleared");
}

// MARK: - Benchmark Implementation Following Android llm_session.cpp

/**
 * Initialize benchmark result structure
 */
- (BenchmarkResultCpp)initializeBenchmarkResult:(int)nPrompt nGenerate:(int)nGenerate nRepeat:(int)nRepeat kvCache:(bool)kvCache {
    BenchmarkResultCpp result;
    result.prompt_tokens = nPrompt;
    result.generate_tokens = nGenerate;
    result.repeat_count = nRepeat;
    result.kv_cache_enabled = kvCache;
    result.success = false;
    return result;
}

/**
 * Initialize LLM for benchmark and verify it's ready
 */
- (BOOL)initializeLlmForBenchmark:(BenchmarkResultCpp&)result callback:(const BenchmarkCallback&)callback {
    if (!_llm) {
        result.error_message = "LLM object is not initialized";
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    // Verify LLM context is valid before proceeding
    auto context = _llm->getContext();
    if (!context) {
        result.error_message = "LLM context is not valid - model may not be properly loaded";
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    // Clear chat history for clean benchmark
    [self clearChatHistory];
    
    // Re-verify context after reset
    context = _llm->getContext();
    if (!context) {
        result.error_message = "LLM context became invalid after reset";
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    return YES;
}

/**
 * Report benchmark progress
 */
- (void)reportBenchmarkProgress:(int)iteration nRepeat:(int)nRepeat nPrompt:(int)nPrompt nGenerate:(int)nGenerate callback:(const BenchmarkCallback&)callback {
    if (callback.onProgress) {
        BenchmarkProgressInfoCpp progressInfo;
        
        if (iteration == 0) {
            progressInfo.progress = 0;
            progressInfo.statusMessage = "Warming up...";
            progressInfo.progressType = 2; // BenchmarkProgressTypeWarmingUp
        } else {
            progressInfo.progress = (iteration * 100) / nRepeat;
            progressInfo.statusMessage = "Running test " + std::to_string(iteration) + "/" + std::to_string(nRepeat) +
                                       " (prompt=" + std::to_string(nPrompt) + ", generate=" + std::to_string(nGenerate) + ")";
            progressInfo.progressType = 3; // BenchmarkProgressTypeRunningTest
        }
        
        // Set structured data
        progressInfo.currentIteration = iteration;
        progressInfo.totalIterations = nRepeat;
        progressInfo.nPrompt = nPrompt;
        progressInfo.nGenerate = nGenerate;
        
        callback.onProgress(progressInfo);
    }
}

/**
 * Run KV cache test iteration
 */
- (BOOL)runKvCacheTest:(int)iteration nPrompt:(int)nPrompt nGenerate:(int)nGenerate
             startTime:(std::chrono::high_resolution_clock::time_point)start_time
                result:(BenchmarkResultCpp&)result callback:(const BenchmarkCallback&)callback {
    
    const int tok = 16; // Same token ID as used in Android llm_session.cpp
    std::vector<int> tokens(nPrompt, tok);
    
    // Validate token vector
    if (tokens.empty() || nPrompt <= 0) {
        result.error_message = "Invalid token configuration for kv-cache test";
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    _llm->response(tokens, nullptr, nullptr, nGenerate);
    
    // Re-get context after response to ensure it's still valid
    auto context = _llm->getContext();
    if (!context) {
        result.error_message = "Context became invalid after response in kv-cache test " + std::to_string(iteration);
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    if (iteration > 0) { // Exclude the first performance value
        auto end_time = std::chrono::high_resolution_clock::now();
        [self processBenchmarkResults:context->prefill_us decodeTime:context->decode_us
                            startTime:start_time endTime:end_time iteration:iteration
                              nPrompt:nPrompt nGenerate:nGenerate result:result
                             callback:callback isKvCache:true];
    }
    return YES;
}

/**
 * Run llama-bench test iteration (without kv cache)
 */
- (BOOL)runLlamaBenchTest:(int)iteration nPrompt:(int)nPrompt nGenerate:(int)nGenerate
                startTime:(std::chrono::high_resolution_clock::time_point)start_time
                   result:(BenchmarkResultCpp&)result callback:(const BenchmarkCallback&)callback {
    
    const int tok = 500;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    std::vector<int> tokens(nPrompt, tok);
    std::vector<int> tokens1(1, tok);
    
    // Validate token vectors
    if ((nPrompt > 0 && tokens.empty()) || tokens1.empty()) {
        result.error_message = "Invalid token configuration for llama-bench test " + std::to_string(iteration);
        if (callback.onError) callback.onError(result.error_message);
        return NO;
    }
    
    NSLog(@"runLlamaBenchTest nPrompt:%d, nGenerate:%d", nPrompt, nGenerate);
    
    if (nPrompt > 0) {
        NSLog(@"runLlamaBenchTest prefill begin");
        _llm->response(tokens, nullptr, nullptr, 1);
        NSLog(@"runLlamaBenchTest prefill end");
        
        auto context = _llm->getContext();
        if (!context) {
            result.error_message = "Context became invalid after prefill response in llama-bench test " + std::to_string(iteration);
            if (callback.onError) callback.onError(result.error_message);
            return NO;
        }
        prefill_us = context->prefill_us;
    }
    
    if (nGenerate > 0) {
        NSLog(@"runLlamaBenchTest generate begin");
        _llm->response(tokens1, nullptr, nullptr, nGenerate);
        NSLog(@"runLlamaBenchTest generate end");
        
        auto context = _llm->getContext();
        if (!context) {
            result.error_message = "Context became invalid after decode response in llama-bench test " + std::to_string(iteration);
            if (callback.onError) callback.onError(result.error_message);
            return NO;
        }
        decode_us = context->decode_us;
    }
    
    if (iteration > 0) { // Exclude the first performance value
        auto end_time = std::chrono::high_resolution_clock::now();
        
        [self processBenchmarkResults:prefill_us decodeTime:decode_us
                            startTime:start_time endTime:end_time iteration:iteration
                              nPrompt:nPrompt nGenerate:nGenerate result:result
                             callback:callback isKvCache:false];
        
        result.sample_times_us.push_back(prefill_us + decode_us);
        result.decode_times_us.push_back(decode_us);
        result.prefill_times_us.push_back(prefill_us);
    }
    return YES;
}

/**
 * Process and report benchmark results
 */
- (void)processBenchmarkResults:(int64_t)prefillTime decodeTime:(int64_t)decodeTime
                      startTime:(std::chrono::high_resolution_clock::time_point)start_time
                        endTime:(std::chrono::high_resolution_clock::time_point)end_time
                      iteration:(int)iteration nPrompt:(int)nPrompt nGenerate:(int)nGenerate
                         result:(BenchmarkResultCpp&)result callback:(const BenchmarkCallback&)callback
                       isKvCache:(bool)isKvCache {
    
    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (isKvCache) {
        result.prefill_times_us.push_back(prefillTime);
        result.decode_times_us.push_back(decodeTime);
    }
    
    // Convert times to seconds
    float runTimeSeconds = runTime / 1000000.0f;
    float prefillTimeSeconds = prefillTime / 1000000.0f;
    float decodeTimeSeconds = decodeTime / 1000000.0f;
    
    // Calculate speeds (tokens per second)
    float prefillSpeed = (prefillTime > 0 && nPrompt > 0) ? ((float)nPrompt / prefillTimeSeconds) : 0.0f;
    float decodeSpeed = (decodeTime > 0 && nGenerate > 0) ? ((float)nGenerate / decodeTimeSeconds) : 0.0f;
    
    // Report detailed results with structured data
    BenchmarkProgressInfoCpp detailedInfo;
    detailedInfo.progress = (iteration * 100) / result.repeat_count;
    detailedInfo.progressType = 3; // BenchmarkProgressTypeRunningTest
    detailedInfo.currentIteration = iteration;
    detailedInfo.totalIterations = result.repeat_count;
    detailedInfo.nPrompt = nPrompt;
    detailedInfo.nGenerate = nGenerate;
    detailedInfo.runTimeSeconds = runTimeSeconds;
    detailedInfo.prefillTimeSeconds = prefillTimeSeconds;
    detailedInfo.decodeTimeSeconds = decodeTimeSeconds;
    detailedInfo.prefillSpeed = prefillSpeed;
    detailedInfo.decodeSpeed = decodeSpeed;
    
    // Format detailed message
    char detailedMsg[1024];
    snprintf(detailedMsg, sizeof(detailedMsg),
        "BenchmarkService: Native Progress [%dp+%dg] (%d%%): Running test %d/%d (prompt=%d, generate=%d) runTime:%.3fs, prefillTime:%.3fs, decodeTime:%.3fs, prefillSpeed:%.2f tok/s, decodeSpeed:%.2f tok/s",
        nPrompt, nGenerate, detailedInfo.progress, iteration, result.repeat_count, nPrompt, nGenerate,
        runTimeSeconds, prefillTimeSeconds, decodeTimeSeconds, prefillSpeed, decodeSpeed);
    
    detailedInfo.statusMessage = std::string(detailedMsg);
    
    NSLog(@"%s", detailedMsg);
    
    if (callback.onProgress) {
        callback.onProgress(detailedInfo);
    }
    
    if (callback.onIterationComplete) {
        callback.onIterationComplete(std::string(detailedMsg));
    }
}

/**
 * Core benchmark implementation
 */
- (BenchmarkResultCpp)runBenchmarkCore:(int)backend threads:(int)threads useMmap:(bool)useMmap power:(int)power
                             precision:(int)precision memory:(int)memory dynamicOption:(int)dynamicOption
                               nPrompt:(int)nPrompt nGenerate:(int)nGenerate nRepeat:(int)nRepeat
                               kvCache:(bool)kvCache callback:(const BenchmarkCallback&)callback {
    
    NSLog(@"BENCHMARK: runBenchmark() STARTED!");
    NSLog(@"BENCHMARK: Parameters - nPrompt=%d, nGenerate=%d, nRepeat=%d, kvCache=%s",
          nPrompt, nGenerate, nRepeat, kvCache ? "true" : "false");
    
    // Initialize result structure
    NSLog(@"BENCHMARK: Initializing benchmark result structure");
    BenchmarkResultCpp result = [self initializeBenchmarkResult:nPrompt nGenerate:nGenerate nRepeat:nRepeat kvCache:kvCache];
    
    // Initialize LLM for benchmark
    NSLog(@"BENCHMARK: About to initialize LLM for benchmark");
    if (![self initializeLlmForBenchmark:result callback:callback]) {
        NSLog(@"BENCHMARK: initializeLlmForBenchmark FAILED!");
        return result;
    }
    NSLog(@"BENCHMARK: initializeLlmForBenchmark SUCCESS - entering benchmark loop");
    
    // Run benchmark iterations
    NSLog(@"BENCHMARK: Starting benchmark loop for %d iterations", nRepeat + 1);
    for (int i = 0; i < nRepeat + 1; ++i) {
        if (_shouldStopBenchmark.load()) {
            result.error_message = "Benchmark stopped by user";
            if (callback.onError) callback.onError(result.error_message);
            return result;
        }
        
        NSLog(@"BENCHMARK: Starting iteration %d/%d", i, nRepeat);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Report progress
        NSLog(@"BENCHMARK: Reporting progress for iteration %d", i);
        [self reportBenchmarkProgress:i nRepeat:nRepeat nPrompt:nPrompt nGenerate:nGenerate callback:callback];
        
        // Run the actual test
        BOOL success;
        if (kvCache) {
            success = [self runKvCacheTest:i nPrompt:nPrompt nGenerate:nGenerate startTime:start_time result:result callback:callback];
        } else {
            success = [self runLlamaBenchTest:i nPrompt:nPrompt nGenerate:nGenerate startTime:start_time result:result callback:callback];
        }
        
        if (!success) {
            return result;
        }
    }
    
    // Report completion
    if (callback.onProgress) {
        BenchmarkProgressInfoCpp completionInfo;
        completionInfo.progress = 100;
        completionInfo.statusMessage = "Benchmark completed!";
        completionInfo.progressType = 5; // BenchmarkProgressTypeCompleted
        callback.onProgress(completionInfo);
    }
    
    result.success = true;
    return result;
}

/**
 * Convert C++ BenchmarkProgressInfoCpp to Objective-C BenchmarkProgressInfo
 */
- (BenchmarkProgressInfo *)convertProgressInfo:(const BenchmarkProgressInfoCpp&)cppInfo {
    BenchmarkProgressInfo *objcInfo = [[BenchmarkProgressInfo alloc] init];
    objcInfo.progress = cppInfo.progress;
    objcInfo.statusMessage = [NSString stringWithUTF8String:cppInfo.statusMessage.c_str()];
    objcInfo.progressType = (BenchmarkProgressType)cppInfo.progressType;
    objcInfo.currentIteration = cppInfo.currentIteration;
    objcInfo.totalIterations = cppInfo.totalIterations;
    objcInfo.nPrompt = cppInfo.nPrompt;
    objcInfo.nGenerate = cppInfo.nGenerate;
    objcInfo.runTimeSeconds = cppInfo.runTimeSeconds;
    objcInfo.prefillTimeSeconds = cppInfo.prefillTimeSeconds;
    objcInfo.decodeTimeSeconds = cppInfo.decodeTimeSeconds;
    objcInfo.prefillSpeed = cppInfo.prefillSpeed;
    objcInfo.decodeSpeed = cppInfo.decodeSpeed;
    return objcInfo;
}

/**
 * Convert C++ BenchmarkResultCpp to Objective-C BenchmarkResult
 */
- (BenchmarkResult *)convertBenchmarkResult:(const BenchmarkResultCpp&)cppResult {
    BenchmarkResult *objcResult = [[BenchmarkResult alloc] init];
    objcResult.success = cppResult.success;
    if (!cppResult.error_message.empty()) {
        objcResult.errorMessage = [NSString stringWithUTF8String:cppResult.error_message.c_str()];
    }
    
    // Convert timing arrays
    NSMutableArray<NSNumber *> *prefillTimes = [[NSMutableArray alloc] init];
    for (int64_t time : cppResult.prefill_times_us) {
        [prefillTimes addObject:@(time)];
    }
    objcResult.prefillTimesUs = [prefillTimes copy];
    
    NSMutableArray<NSNumber *> *decodeTimes = [[NSMutableArray alloc] init];
    for (int64_t time : cppResult.decode_times_us) {
        [decodeTimes addObject:@(time)];
    }
    objcResult.decodeTimesUs = [decodeTimes copy];
    
    NSMutableArray<NSNumber *> *sampleTimes = [[NSMutableArray alloc] init];
    for (int64_t time : cppResult.sample_times_us) {
        [sampleTimes addObject:@(time)];
    }
    objcResult.sampleTimesUs = [sampleTimes copy];
    
    objcResult.promptTokens = cppResult.prompt_tokens;
    objcResult.generateTokens = cppResult.generate_tokens;
    objcResult.repeatCount = cppResult.repeat_count;
    objcResult.kvCacheEnabled = cppResult.kv_cache_enabled;
    
    return objcResult;
}

// MARK: - Public Benchmark Methods

/**
 * Run official benchmark following llm_bench.cpp approach
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
                       completeCallback:(BenchmarkCompleteCallback _Nullable)completeCallback {
    
    if (_isBenchmarkRunning.load()) {
        if (errorCallback) {
            errorCallback(@"Benchmark is already running");
        }
        return;
    }
    
    if (!_llm) {
        if (errorCallback) {
            errorCallback(@"Model is not initialized");
        }
        return;
    }
    
    _isBenchmarkRunning = true;
    _shouldStopBenchmark = false;
    
    // Run benchmark in background thread
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        @try {
            // Create C++ callback structure
            BenchmarkCallback cppCallback;
            
            cppCallback.onProgress = [progressCallback, self](const BenchmarkProgressInfoCpp& progressInfo) {
                if (progressCallback) {
                    BenchmarkProgressInfo *objcProgressInfo = [self convertProgressInfo:progressInfo];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        progressCallback(objcProgressInfo);
                    });
                }
            };
            
            cppCallback.onError = [errorCallback](const std::string& error) {
                if (errorCallback) {
                    NSString *errorStr = [NSString stringWithUTF8String:error.c_str()];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        errorCallback(errorStr);
                    });
                }
            };
            
            cppCallback.onIterationComplete = [iterationCompleteCallback](const std::string& detailed_stats) {
                if (iterationCompleteCallback) {
                    NSString *statsStr = [NSString stringWithUTF8String:detailed_stats.c_str()];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        iterationCompleteCallback(statsStr);
                    });
                }
            };
            
            // Run the actual benchmark
            BenchmarkResultCpp cppResult = [self runBenchmarkCore:(int)backend
                                                          threads:(int)threads
                                                          useMmap:(bool)useMmap
                                                            power:(int)power
                                                        precision:(int)precision
                                                           memory:(int)memory
                                                    dynamicOption:(int)dynamicOption
                                                          nPrompt:(int)nPrompt
                                                        nGenerate:(int)nGenerate
                                                          nRepeat:(int)nRepeat
                                                          kvCache:(bool)kvCache
                                                         callback:cppCallback];
            
            // Convert result and call completion callback
            BenchmarkResult *objcResult = [self convertBenchmarkResult:cppResult];
            
            dispatch_async(dispatch_get_main_queue(), ^{
                if (completeCallback) {
                    completeCallback(objcResult);
                }
            });
            
        }
        @catch (NSException *exception) {
            NSLog(@"Exception during benchmark: %@", exception.reason);
            if (errorCallback) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    errorCallback([NSString stringWithFormat:@"Benchmark failed: %@", exception.reason]);
                });
            }
        }
        @finally {
            self->_isBenchmarkRunning = false;
        }
    });
}

/**
 * Stop running benchmark
 */
- (void)stopBenchmark {
    _shouldStopBenchmark = true;
    NSLog(@"Benchmark stop requested");
}

/**
 * Check if benchmark is currently running
 */
- (BOOL)isBenchmarkRunning {
    return _isBenchmarkRunning.load();
}

@end
