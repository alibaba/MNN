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

+ (NSString *)defaultCfgPrompt {
    return @"Generate an image.";
}

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

/// Processes text prompts through the LLM to generate embeddings for diffusion.
///
/// This method performs the following steps:
/// 1. Formats prompts using instruction and generator templates
/// 2. Tokenizes and pads the prompts to equal length
/// 3. Generates embeddings using the LLM
/// 4. Concatenates meta queries with the embeddings
/// 5. Builds attention mask and position IDs
/// 6. Forwards through LLM to get hidden states
/// 7. Extracts the last 256 tokens as output embeddings
///
/// @param prompt The main style prompt.
/// @param cfgPrompt The classifier-free guidance prompt.
/// @return VARP containing the processed embeddings, or nullptr on failure.
- (VARP)processPrompt:(NSString *)prompt cfgPrompt:(NSString *)cfgPrompt {
    if (mMetaQueries.empty()) {
        return nullptr;
    }
    
    // Define prompt formatting templates
    std::string img_start_token = "<|vision_bos|>";
    std::string instruction_fmt = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n";
    std::string generator_fmt = "Generate an image: {input}";
    
    // Lambda to format a prompt using the templates
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
    
    // Format both prompts
    std::vector<std::string> prompts;
    prompts.push_back(format_prompt([prompt UTF8String]));
    prompts.push_back(format_prompt([cfgPrompt UTF8String]));
    
    // Tokenize all prompts and find max length
    std::vector<std::vector<int>> all_ids;
    int max_len = 0;
    for (const auto& p : prompts) {
        auto ids = mLlm->tokenizer_encode(p);
        all_ids.push_back(ids);
        if (static_cast<int>(ids.size()) > max_len) {
            max_len = static_cast<int>(ids.size());
        }
    }
    
    // Pad sequences to equal length (right padding)
    int batch = static_cast<int>(all_ids.size());
    std::vector<int> flat_input_ids;
    std::vector<int> flat_mask;
    int pad_id = 151643;  // Qwen default padding token ID
    
    for (const auto& ids : all_ids) {
        int len = static_cast<int>(ids.size());
        int pad_len = max_len - len;
        for (int id : ids) {
            flat_input_ids.push_back(id);
            flat_mask.push_back(1);
        }
        for (int i = 0; i < pad_len; ++i) {
            flat_input_ids.push_back(pad_id);
            flat_mask.push_back(0);
        }
    }
    
    // Generate embeddings from token IDs
    auto inputs_embeds = mLlm->embedding(flat_input_ids);
    inputs_embeds = _Reshape(inputs_embeds, {batch, max_len, -1});
    
    // Prepare meta queries (256 learnable query vectors)
    int num_queries = 256;
    auto meta_queries = mMetaQueries[0];
    auto meta_queries_batch = _Unsqueeze(meta_queries, {0});
    meta_queries_batch = _Concat({meta_queries_batch, meta_queries_batch}, 0);
    
    // Concatenate embeddings with meta queries
    auto full_embeds = _Concat({inputs_embeds, meta_queries_batch}, 1);
    
    // Build causal attention mask
    int total_len = max_len + num_queries;
    std::vector<int> mask_data(batch * total_len * total_len, 0);
    
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < total_len; ++i) {
            for (int j = 0; j < total_len; ++j) {
                // Causal mask: can only attend to previous positions
                if (j > i) continue;
                
                // Check if position j is valid (not a padding token)
                bool valid_j = true;
                if (j < max_len) {
                    if (flat_mask[b * max_len + j] == 0) {
                        valid_j = false;
                    }
                }
                if (valid_j) {
                    mask_data[b * total_len * total_len + i * total_len + j] = 1;
                }
            }
        }
    }
    auto attention_mask = _Const(mask_data.data(), {batch, 1, total_len, total_len}, NCHW, halide_type_of<int>());
    
    // Build position IDs (cumulative sum of attention mask)
    std::vector<int> pos_ids(batch * total_len);
    for (int b = 0; b < batch; ++b) {
        int cumsum = 0;
        for (int i = 0; i < total_len; ++i) {
            int val = 1;
            if (i < max_len) {
                val = flat_mask[b * max_len + i];
            }
            cumsum += val;
            int pos = cumsum - 1;
            if (pos < 0) pos = 0;
            pos_ids[b * total_len + i] = pos;
        }
    }
    auto position_ids = _Const(pos_ids.data(), {batch, total_len}, NCHW, halide_type_of<int>());
    
    // Forward through LLM to get hidden states
    auto outputs = mLlm->forwardRaw(full_embeds, attention_mask, position_ids);
    int hiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    hiddenStateIndex = hiddenStateIndex == -1 ? static_cast<int>(outputs.size()) - 1 : hiddenStateIndex;
    
    auto output = outputs[hiddenStateIndex];
    output = _Reshape(output, {batch, -1, output->getInfo()->dim[2]});
    
    // Extract last 256 tokens as the embedding output
    auto splits = _Split(output, {total_len - num_queries, num_queries}, 1);
    
    return splits[1];
}

#pragma mark - Style Transfer

- (void)runStyleTransferWithInputImage:(NSString *)inputImagePath
                                prompt:(NSString *)prompt
                             cfgPrompt:(NSString *)cfgPrompt
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
            // Stage 1: Process prompt with LLM
            dispatch_async(dispatch_get_main_queue(), ^{
                if (progressCallback) {
                    progressCallback(5, @"Processing prompt...");
                }
            });
            
            NSLog(@"SanaDiffusionSession: Processing prompt: %@", prompt);
            VARP llmOutput = [self processPrompt:prompt cfgPrompt:cfgPrompt];
            
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
            
            bool success = self->mDiffusion->run(
                llmOutput,
                [inputImagePath UTF8String],
                [outputPath UTF8String],
                iterations,
                seed,
                diffusionProgressCallback
            );
            
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
