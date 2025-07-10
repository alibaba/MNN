//
//  LLMInferenceEngineWrapper.h
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

#ifndef LLMInferenceEngineWrapper_h
#define LLMInferenceEngineWrapper_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^CompletionHandler)(BOOL success);
typedef void (^OutputHandler)(NSString * _Nonnull output);

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

@end

NS_ASSUME_NONNULL_END

#endif /* LLMInferenceEngineWrapper_h */