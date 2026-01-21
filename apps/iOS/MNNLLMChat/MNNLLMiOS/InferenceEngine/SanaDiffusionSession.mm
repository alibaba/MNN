//
//  SanaDiffusionSession.mm
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/01/19.
//  Copyright © 2025 MNN. All rights reserved.
//

#import "SanaDiffusionSession.h"

#import <Foundation/Foundation.h>

#include <MNN/llm/llm.hpp>
#include <MNN/diffusion/diffusion.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
//#include <cv/cv.hpp>

#include <random>
#include <algorithm>
#include <fstream>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Transformer;
using namespace MNN::DIFFUSION;
using namespace CV;

#pragma mark - Private Interface

@implementation SanaDiffusionSession {
    /// LLM instance for processing text prompts to embeddings.
    std::unique_ptr<Llm> mLlm;
    
    /// Meta query vectors loaded from the model.
    std::vector<VARP> mMetaQueries;
    
    /// Diffusion model instance using SANA_DIFFUSION type.
    std::shared_ptr<Diffusion> mDiffusion;
    
    /// Path to the model directory.
    NSString *mModelPath;
    
    /// Memory mode for Diffusion model (0 = default).
    int mMemoryMode;
    
    /// Flag indicating whether the model has been loaded.
    BOOL _isModelLoaded;
    
    /// Flag indicating whether a style transfer is in progress.
    BOOL _isProcessing;
}

#pragma mark - Class Methods

+ (NSString *)defaultGhibliPrompt {
    return @"Convert to a Ghibli-style illustration: soft contrast, warm tones, slight linework, keep the scene consistent.";
}

#pragma mark - Initialization

- (instancetype)initWithModelPath:(NSString *)path completion:(SanaCompletionHandler)completion {
    self = [super init];
    if (self) {
        mModelPath = path;
        mMemoryMode = 1;
        _isModelLoaded = NO;
        _isProcessing = NO;
        
        // Load model asynchronously
        // Note: LLM loading can be on background thread, but Diffusion (Metal) must be on main thread
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            // Step 1: Load LLM on background thread (no Metal dependency)
            BOOL llmSuccess = [self loadLLMModel];
            if (!llmSuccess) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_isModelLoaded = NO;
                    if (completion) {
                        completion(NO);
                    }
                });
                return;
            }
            
            // Step 2: Load Diffusion on main thread (Metal requires main thread for UIApplication access)
            dispatch_async(dispatch_get_main_queue(), ^{
                BOOL diffusionSuccess = [self loadDiffusionModel];
                self->_isModelLoaded = diffusionSuccess;
                if (completion) {
                    completion(diffusionSuccess);
                }
            });
        });
    }
    return self;
}

#pragma mark - Model Loading

/// Loads the LLM model from disk. Can be called from background thread.
/// @return YES if LLM loaded successfully, NO otherwise.
- (BOOL)loadLLMModel {
    @try {
        NSLog(@"SanaDiffusionSession: Starting model loading from %@", mModelPath);
        
        // Load LLM for prompt processing
        NSString *llmPath = [mModelPath stringByAppendingPathComponent:@"llm"];
        NSString *llmConfigPath = [llmPath stringByAppendingPathComponent:@"config.json"];
        NSLog(@"SanaDiffusionSession: Loading LLM from %@", llmConfigPath);
        
        mLlm.reset(Llm::createLLM([llmConfigPath UTF8String]));
        if (!mLlm) {
            NSLog(@"SanaDiffusionSession: Failed to create LLM");
            return NO;
        }
        
        // Enable hidden states output for embedding extraction
        mLlm->set_config("{\"hidden_states\":true}");
        mLlm->load();
        
        // Load meta queries for LLM prompt formatting
        NSString *metaPath = [llmPath stringByAppendingPathComponent:@"meta_queries.mnn"];
        NSLog(@"SanaDiffusionSession: Loading meta queries from %@", metaPath);
        mMetaQueries = Variable::load([metaPath UTF8String]);
        if (!mMetaQueries.empty()) {
            mMetaQueries[0].fix(VARP::CONSTANT);
        } else {
            NSLog(@"SanaDiffusionSession: Failed to load meta_queries.mnn");
            return NO;
        }
        
        NSLog(@"SanaDiffusionSession: LLM loading complete");
        return YES;
        
    } @catch (NSException *exception) {
        NSLog(@"SanaDiffusionSession: Exception during LLM loading: %@", exception.reason);
        return NO;
    }
}

/// Loads the Diffusion model from disk. Must be called from main thread (Metal requirement).
/// @return YES if Diffusion loaded successfully, NO otherwise.
- (BOOL)loadDiffusionModel {
    @try {
        // Load Diffusion model with SANA_DIFFUSION type
        // Note: Must be on main thread because Metal backend calls [UIApplication applicationState]
        NSLog(@"SanaDiffusionSession: Loading Sana Diffusion model");
        Diffusion* rawDiffusion = Diffusion::createDiffusion(
            [mModelPath UTF8String],
            DiffusionModelType::SANA_DIFFUSION,
            MNNForwardType::MNN_FORWARD_METAL,
            mMemoryMode
        );
        
        if (!rawDiffusion) {
            NSLog(@"SanaDiffusionSession: Failed to create Diffusion");
            return NO;
        }
        
        mDiffusion = std::shared_ptr<Diffusion>(rawDiffusion);
        if (!mDiffusion->load()) {
            NSLog(@"SanaDiffusionSession: Failed to load Diffusion model");
            return NO;
        }
        
        NSLog(@"SanaDiffusionSession: Model loading complete");
        return YES;
        
    } @catch (NSException *exception) {
        NSLog(@"SanaDiffusionSession: Exception during Diffusion loading: %@", exception.reason);
        return NO;
    }
}

#pragma mark - LLM Processing

/// Processes a single text prompt through the LLM to generate embeddings for diffusion.
/// Based on sana_llm.hpp implementation - processes batch=1.
///
/// @param prompt The text prompt to process.
/// @return VARP containing the processed embeddings [1, 256, hidden_size], or nullptr on failure.
- (VARP)processSinglePrompt:(NSString *)prompt {
    if (mMetaQueries.empty()) {
        NSLog(@"SanaDiffusionSession: ERROR - mMetaQueries is empty!");
        return nullptr;
    }
    
    // Define prompt formatting templates (same as sana_llm.hpp)
    std::string img_start_token = "<|vision_bos|>";
    std::string instruction_fmt = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n";
    std::string generator_fmt = "Generate an image: {input}";
    
    // Format the prompt
    auto format_prompt = [&](const std::string& input) {
        std::string text = generator_fmt;
        std::string placeholder = "{input}";
        size_t pos = text.find(placeholder);
        if (pos != std::string::npos) {
            text.replace(pos, placeholder.length(), input);
        }
        std::string full = instruction_fmt;
        pos = full.find(placeholder);
        if (pos != std::string::npos) {
            full.replace(pos, placeholder.length(), text);
        }
        return full + img_start_token;
    };
    
    std::string formatted_prompt = format_prompt([prompt UTF8String]);
    
    // Tokenize
    auto input_ids = mLlm->tokenizer_encode(formatted_prompt);
    int seq_len = static_cast<int>(input_ids.size());
    int batch = 1;
    
    NSLog(@"SanaDiffusionSession: processSinglePrompt - seq_len=%d", seq_len);
    
    // Generate embeddings from token IDs
    auto inputs_embeds = mLlm->embedding(input_ids);
    inputs_embeds = _Reshape(inputs_embeds, {batch, seq_len, -1});
    
    // Prepare meta queries [256, hidden_size] -> [1, 256, hidden_size]
    int num_queries = 256;
    auto meta_queries = mMetaQueries[0];
    auto meta_queries_batch = _Unsqueeze(meta_queries, {0}); // [1, 256, H]
    
    // Concatenate on sequence dimension: [1, seq_len, H] + [1, 256, H] -> [1, seq_len+256, H]
    auto full_embeds = _Concat({inputs_embeds, meta_queries_batch}, 1);
    
    int total_len = seq_len + num_queries;
    
    // Build causal attention mask [1, 1, total_len, total_len]
    std::vector<int> mask_data(total_len * total_len, 0);
    for (int i = 0; i < total_len; ++i) {
        for (int j = 0; j <= i; ++j) {
            mask_data[i * total_len + j] = 1;
        }
    }
    auto attention_mask = _Const(mask_data.data(), {batch, 1, total_len, total_len}, NCHW, halide_type_of<int>());
    
    // Build position IDs [1, total_len]
    std::vector<int> pos_ids(total_len);
    for (int i = 0; i < total_len; ++i) {
        pos_ids[i] = i;
    }
    auto position_ids = _Const(pos_ids.data(), {batch, total_len}, NCHW, halide_type_of<int>());
    
    // Reset LLM state before forward
    mLlm->reset();
    
    // Forward through LLM
    auto outputs = mLlm->forwardRaw(full_embeds, attention_mask, position_ids);
    int hiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    hiddenStateIndex = hiddenStateIndex == -1 ? static_cast<int>(outputs.size()) - 1 : hiddenStateIndex;
    
    auto output = outputs[hiddenStateIndex];
    output = _Reshape(output, {batch, -1, output->getInfo()->dim[2]});
    
    // Extract last 256 tokens (the query embeddings)
    auto splits = _Split(output, {total_len - num_queries, num_queries}, 1);
    
    auto result = splits[1]; // [1, 256, hidden_size]
    
    // Debug output
    auto resultInfo = result->getInfo();
    if (resultInfo) {
        NSLog(@"SanaDiffusionSession: processSinglePrompt output shape: [%d, %d, %d]",
              resultInfo->dim[0], resultInfo->dim[1], resultInfo->dim[2]);
    }
    
    return result;
}

#pragma mark - Style Transfer

- (void)runStyleTransferWithInputImage:(NSString *)inputImagePath
                                prompt:(NSString *)prompt
                            outputPath:(NSString *)outputPath
                            iterations:(int)iterations
                                  seed:(int)seed
                      progressCallback:(SanaProgressHandler)progressCallback
                            completion:(SanaStyleTransferCompletion)completion {
    
    // Validate model state
    if (!_isModelLoaded) {
        if (completion) {
            completion(NO, @"Model not loaded");
        }
        return;
    }
    
    if (_isProcessing) {
        if (completion) {
            completion(NO, @"Already processing");
        }
        return;
    }
    
    _isProcessing = YES;
    
    // Process on background thread
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            // Record total start time
            NSDate *totalStartTime = [NSDate date];
            NSDate *llmStartTime = [NSDate date];
            
            // Stage 1: Process prompt with LLM
            dispatch_async(dispatch_get_main_queue(), ^{
                if (progressCallback) {
                    progressCallback(5, @"Processing prompt...");
                }
            });
            
            NSLog(@"SanaDiffusionSession: Processing prompt: %@", prompt);
            VARP llmOutput = [self processSinglePrompt:prompt];
            
            NSTimeInterval llmDuration = [[NSDate date] timeIntervalSinceDate:llmStartTime] * 1000;
            NSLog(@"SanaDiffusionSession: LLM processing time: %.2f ms", llmDuration);
            
            if (llmOutput.get() == nullptr) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self->_isProcessing = NO;
                    if (completion) {
                        completion(NO, @"LLM processing failed");
                    }
                });
                return;
            }
            
            dispatch_async(dispatch_get_main_queue(), ^{
                if (progressCallback) {
                    progressCallback(15, @"Running diffusion...");
                }
            });
            
            // Stage 2: Run diffusion pipeline with LLM embeddings
            NSDate *diffusionStartTime = [NSDate date];
            
            auto diffusionProgressCallback = [progressCallback](int step) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (progressCallback) {
                        // Map diffusion progress (0-100) to overall range (15-95)
                        int progress = 15 + (step * 80 / 100);
                        NSString *stage = [NSString stringWithFormat:@"Diffusion step %d%%", step];
                        progressCallback(progress, stage);
                    }
                });
            };
            
            // Debug: Print llmOutput tensor info
            auto llmOutputInfo = llmOutput->getInfo();
            if (llmOutputInfo) {
                NSMutableString *dimStr = [NSMutableString stringWithString:@"["];
                for (size_t i = 0; i < llmOutputInfo->dim.size(); i++) {
                    [dimStr appendFormat:@"%d", llmOutputInfo->dim[i]];
                    if (i < llmOutputInfo->dim.size() - 1) {
                        [dimStr appendString:@", "];
                    }
                }
                [dimStr appendString:@"]"];
                NSLog(@"SanaDiffusionSession: llmOutput shape: %@, size: %d, order: %d, type: %d",
                      dimStr,
                      llmOutputInfo->size,
                      (int)llmOutputInfo->order,
                      (int)llmOutputInfo->type.code);
            } else {
                NSLog(@"SanaDiffusionSession: llmOutput info is NULL!");
            }
            
            bool success = self->mDiffusion->run(
                llmOutput,
                [inputImagePath UTF8String],
                [outputPath UTF8String],
                iterations,
                seed,
                diffusionProgressCallback
            );
            
            NSTimeInterval diffusionDuration = [[NSDate date] timeIntervalSinceDate:diffusionStartTime] * 1000;
            NSTimeInterval totalDuration = [[NSDate date] timeIntervalSinceDate:totalStartTime] * 1000;
            
            NSLog(@"SanaDiffusionSession: Diffusion time: %.2f ms", diffusionDuration);
            NSLog(@"SanaDiffusionSession: Total time: %.2f ms", totalDuration);
            
            // Save benchmark results to file
            [self saveBenchmarkResult:@{
                @"timestamp": [NSDate date],
                @"iterations": @(iterations),
                @"seed": @(seed),
                @"llm_time_ms": @(llmDuration),
                @"diffusion_time_ms": @(diffusionDuration),
                @"total_time_ms": @(totalDuration),
                @"success": @(success),
                @"input_image": inputImagePath,
                @"output_image": outputPath
            }];
            
            // Report completion on main thread
            dispatch_async(dispatch_get_main_queue(), ^{
                self->_isProcessing = NO;
                
                if (progressCallback) {
                    progressCallback(100, success ? @"Complete!" : @"Failed");
                }
                
                if (completion) {
                    if (success) {
                        completion(YES, nil);
                    } else {
                        completion(NO, @"Diffusion processing failed");
                    }
                }
            });
            
        } @catch (NSException *exception) {
            dispatch_async(dispatch_get_main_queue(), ^{
                self->_isProcessing = NO;
                if (completion) {
                    completion(NO, [NSString stringWithFormat:@"Exception: %@", exception.reason]);
                }
            });
        }
    });
}

#pragma mark - Benchmark

/// Saves benchmark result to a JSON file in the Documents directory.
/// @param result Dictionary containing benchmark metrics.
- (void)saveBenchmarkResult:(NSDictionary *)result {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    NSString *benchmarkFile = [documentsDirectory stringByAppendingPathComponent:@"sana_diffusion_benchmark.json"];
    
    // Read existing results or create new array
    NSMutableArray *results = [NSMutableArray array];
    if ([fileManager fileExistsAtPath:benchmarkFile]) {
        NSData *existingData = [NSData dataWithContentsOfFile:benchmarkFile];
        if (existingData) {
            NSError *error = nil;
            NSArray *existingResults = [NSJSONSerialization JSONObjectWithData:existingData options:0 error:&error];
            if (existingResults && !error) {
                [results addObjectsFromArray:existingResults];
            }
        }
    }
    
    // Format timestamp as string
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy-MM-dd HH:mm:ss"];
    NSMutableDictionary *formattedResult = [result mutableCopy];
    if (result[@"timestamp"]) {
        formattedResult[@"timestamp"] = [formatter stringFromDate:result[@"timestamp"]];
    }
    
    // Add new result
    [results addObject:formattedResult];
    
    // Write back to file
    NSError *writeError = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:results options:NSJSONWritingPrettyPrinted error:&writeError];
    if (jsonData && !writeError) {
        [jsonData writeToFile:benchmarkFile atomically:YES];
        NSLog(@"SanaDiffusionSession: Benchmark saved to %@", benchmarkFile);
    } else {
        NSLog(@"SanaDiffusionSession: Failed to save benchmark: %@", writeError);
    }
}

/// Returns the path to the benchmark results file.
+ (NSString *)benchmarkFilePath {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    return [documentsDirectory stringByAppendingPathComponent:@"sana_diffusion_benchmark.json"];
}

#pragma mark - Properties

- (BOOL)isModelLoaded {
    return _isModelLoaded;
}

- (BOOL)isProcessing {
    return _isProcessing;
}

#pragma mark - Memory Management

- (void)dealloc {
    mLlm.reset();
    mMetaQueries.clear();
    mDiffusion.reset();
    NSLog(@"SanaDiffusionSession deallocated");
}

@end
