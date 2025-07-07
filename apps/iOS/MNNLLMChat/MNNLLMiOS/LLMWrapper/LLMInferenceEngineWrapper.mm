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
#include <MNN/llm/llm.hpp>
#include <vector>
#include <utility>

#import <Foundation/Foundation.h>
#import "LLMInferenceEngineWrapper.h"

using namespace MNN::Transformer;

using ChatMessage = std::pair<std::string, std::string>;

@implementation LLMInferenceEngineWrapper {
    std::shared_ptr<Llm> _llm;
    std::vector<ChatMessage> _history;
    std::mutex _historyMutex;
    std::atomic<bool> _isProcessing;
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
    try {
        if (std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        NSLog(@"Error removing directory %s: %s", path.c_str(), e.what());
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
        
        _llm.reset(Llm::createLLM(config_path));
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
        
        // Get memory mapping setting with default fallback
        BOOL useMmap = configDict[@"use_mmap"] == nil ? YES : [configDict[@"use_mmap"] boolValue];
        
        // Create LLM instance with error checking
        _llm.reset(Llm::createLLM(config_path));
        if (!_llm) {
            NSLog(@"Error: Failed to create LLM instance from config: %s", config_path.c_str());
            return NO;
        }
        
        // Setup temporary directory with improved error handling
        std::string model_path_str([modelPath UTF8String]);
        std::string temp_directory_path = model_path_str + "/temp";
        
        // Clean up existing temp directory
        if (!remove_directory_safely(temp_directory_path)) {
            NSLog(@"Warning: Failed to remove existing temp directory, continuing...");
        }
        
        // Create new temp directory
        if (mkdir(temp_directory_path.c_str(), 0755) != 0 && errno != EEXIST) {
            NSLog(@"Error: Failed to create temp directory: %s, errno: %d", temp_directory_path.c_str(), errno);
            return NO;
        }
        
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
//            lastChar == ' ' ||
            lastChar == '\t' ||
            lastChar == '.' ||
            lastChar == ',' ||
            lastChar == ';' ||
            lastChar == ':' ||
            lastChar == '!' ||
            lastChar == '?'
//            lastChar == '"' ||
//            lastChar == '\'' ||
//            lastChar == ')' ||
//            lastChar == ']' ||
//            lastChar == '}' ||
//            lastChar == '>' ||
//            lastChar == '-' ||
//            lastChar == '_' ||
//            lastChar == '~' ||
//            lastChar == '`'
            ) {
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
//                "\xE2\x80\x9C",     // "
//                "\xE2\x80\x9D",     // "
//                "\xEF\xBC\x89",     // ）
//                "\xE3\x80\x91",     // 】
//                "\xE3\x80\x8B",     // 》
                "\xE2\x80\xA6",     // …
//                "\xE2\x80\x93",     // –
//                "\xE2\x80\x94",     // —
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
    
    _isProcessing = true;
    
    // Use high priority queue for better responsiveness
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        @try {
            auto inference_start_time = std::chrono::high_resolution_clock::now();
            
            OptimizedLlmStreamBuffer::CallBack callback = [output, self](const char* str, size_t len) {
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
                std::lock_guard<std::mutex> lock(self->_historyMutex);
                self->_history.emplace_back(ChatMessage("user", [input UTF8String]));
            }
            
            std::string inputStr = [input UTF8String];
            if (inputStr == "benchmark") {
                [self performBenchmarkWithOutput:&os];
            } else {
                // Get initial context state for performance measurement
                auto context = self->_llm->getContext();
                int initial_prompt_len = context->prompt_len;
                int initial_decode_len = context->gen_seq_len;
                int64_t initial_prefill_time = context->prefill_us;
                int64_t initial_decode_time = context->decode_us;
                
                // Execute inference
                self->_llm->response(self->_history, &os, "<eop>", 999999);
                
                // Calculate performance metrics if requested
                if (showPerformance) {
                    auto inference_end_time = std::chrono::high_resolution_clock::now();
                    auto total_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time);
                    
                    // Get final context state
                    int final_prompt_len = context->prompt_len;
                    int final_decode_len = context->gen_seq_len;
                    int64_t final_prefill_time = context->prefill_us;
                    int64_t final_decode_time = context->decode_us;
                    
                    // Calculate differences for this inference
                    int current_prompt_len = final_prompt_len - initial_prompt_len;
                    int current_decode_len = final_decode_len - initial_decode_len;
                    int64_t current_prefill_time = final_prefill_time - initial_prefill_time;
                    int64_t current_decode_time = final_decode_time - initial_decode_time;
                    
                    float prefill_s = current_prefill_time / 1e6;
                    float decode_s = current_decode_time / 1e6;
                    
                    // Format performance results
                    std::ostringstream performance_output;
                    performance_output << "\n\n> Performance Results:\n"
                                      << "> Total inference time: " << total_inference_time.count() << " ms\n;"
                                      << "> Prompt tokens: " << current_prompt_len << "\n;"
                                      << "> Generated tokens: " << current_decode_len << "\n;"
                                      << "> Prefill time: " << std::fixed << std::setprecision(2) << prefill_s << " s\n;"
                                      << "> Decode time: " << std::fixed << std::setprecision(2) << decode_s << " s\n;"
                                      << "> Prefill speed: " << std::fixed << std::setprecision(2)
                                      << (prefill_s > 0 ? current_prompt_len / prefill_s : 0) << " tok/s\n;"
                                      << "> Decode speed: " << std::fixed << std::setprecision(2)
                                      << (decode_s > 0 ? current_decode_len / decode_s : 0) << " tok/s\n\n.";
                    
                    // Output performance results
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
            self->_isProcessing = false;
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
 * Enhanced deallocation with proper cleanup
 */
- (void)dealloc {
    NSLog(@"LLMInferenceEngineWrapper deallocating...");
    
    // Wait for any ongoing processing to complete
    while (_isProcessing.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
 * New method to check if model is ready for inference
 * 
 * @return YES if model is loaded and ready
 */
- (BOOL)isModelReady {
    return _llm != nullptr && !_isProcessing.load();
}

/**
 * New method to get current processing status
 * 
 * @return YES if currently processing an inference request
 */
- (BOOL)isProcessing {
    return _isProcessing.load();
}

/**
 * New method to cancel ongoing inference (if supported)
 */
- (void)cancelInference {
    if (_isProcessing.load()) {
        NSLog(@"Inference cancellation requested");
        // Note: Actual cancellation depends on MNN LLM implementation
        // This is a placeholder for future enhancement
    }
}

/**
 * New method to get chat history count
 * 
 * @return Number of messages in chat history
 */
- (NSUInteger)getChatHistoryCount {
    std::lock_guard<std::mutex> lock(_historyMutex);
    return _history.size();
}

/**
 * New method to clear chat history
 */
- (void)clearChatHistory {
    std::lock_guard<std::mutex> lock(_historyMutex);
    _history.clear();
    NSLog(@"Chat history cleared");
}

@end
