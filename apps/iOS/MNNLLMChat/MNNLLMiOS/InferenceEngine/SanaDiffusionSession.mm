//
//  SanaDiffusionSession.mm
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/01/19.
//  Copyright © 2025 MNN. All rights reserved.
//

#import "SanaDiffusionSession.h"

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#include <MNN/diffusion/diffusion.hpp>
#include <MNN/diffusion/sana_llm.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

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
    /// SanaLlm instance for processing text prompts to embeddings (from sana_llm.hpp).
    std::unique_ptr<SanaLlm> mSanaLlm;
    
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
        
        // Load model asynchronously on background thread
        // Threading strategy: All model loading and inference on background thread
        // to keep UI responsive. Accept UIApplication warning for better UX.
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            // Step 1: Load LLM on background thread
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
            
            // Step 2: Load Diffusion on background thread
            BOOL diffusionSuccess = [self loadDiffusionModel];
            
            dispatch_async(dispatch_get_main_queue(), ^{
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

/// Loads the SanaLlm model from disk using sana_llm.hpp. Can be called from background thread.
/// @return YES if SanaLlm loaded successfully, NO otherwise.
- (BOOL)loadLLMModel {
    @try {
        NSLog(@"SanaDiffusionSession: Starting SanaLlm loading from %@", mModelPath);
        
        // Load SanaLlm using sana_llm.hpp - it handles LLM and meta_queries loading internally
        NSString *llmPath = [mModelPath stringByAppendingPathComponent:@"llm"];
        NSLog(@"SanaDiffusionSession: Loading SanaLlm from %@", llmPath);
        
        mSanaLlm = std::make_unique<SanaLlm>([llmPath UTF8String]);
        if (!mSanaLlm) {
            NSLog(@"SanaDiffusionSession: Failed to create SanaLlm");
            return NO;
        }
        
        NSLog(@"SanaDiffusionSession: SanaLlm loading complete");
        return YES;
        
    } @catch (NSException *exception) {
        NSLog(@"SanaDiffusionSession: Exception during SanaLlm loading: %@", exception.reason);
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

/// Processes a single text prompt through SanaLlm to generate embeddings for diffusion.
/// Uses sana_llm.hpp's SanaLlm::process() method directly.
///
/// @param prompt The text prompt to process.
/// @return VARP containing the processed embeddings [1, 256, hidden_size], or nullptr on failure.
- (VARP)processSinglePrompt:(NSString *)prompt {
    if (!mSanaLlm) {
        NSLog(@"SanaDiffusionSession: ERROR - mSanaLlm is not initialized!");
        return nullptr;
    }
    
    NSLog(@"SanaDiffusionSession: Processing prompt via SanaLlm: %@", prompt);
    
    // Use SanaLlm::process() directly - it handles all the formatting, tokenization, and LLM forward
    VARP result = mSanaLlm->process([prompt UTF8String]);
    
    if (result.get() == nullptr) {
        NSLog(@"SanaDiffusionSession: ERROR - SanaLlm::process() returned nullptr");
        return nullptr;
    }
    
    // Debug output
    auto resultInfo = result->getInfo();
    if (resultInfo) {
        NSLog(@"SanaDiffusionSession: SanaLlm output shape: [%d, %d, %d]",
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
            completion(NO, @"Model not loaded", 0);
        }
        return;
    }
    
    if (_isProcessing) {
        if (completion) {
            completion(NO, @"Already processing", 0);
        }
        return;
    }
    
    _isProcessing = YES;
    
    // Process on background thread to keep UI responsive.
    // Note: This will trigger "[UIApplication applicationState] must be used from main thread only"
    // warning from MNN Metal backend. This is expected and acceptable - see threading strategy
    // comment in initWithModelPath:completion: for details.
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            // Normalize image orientation if needed
            NSString *normalizedInputPath = [self normalizeImageOrientation:inputImagePath];
            if (!normalizedInputPath) {
                normalizedInputPath = inputImagePath;
            }

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
                        completion(NO, @"LLM processing failed", 0);
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
            
            // The pre-built MNN.framework's SanaDiffusion::run() does not invoke the
            // C++ progressCallback during denoising. Work around this by using a
            // dispatch timer on the main queue that fires while run() blocks the
            // background thread, giving the user periodic progress updates.
            __block int timerProgress = 15;
            int progressPerTick = MAX(1, 80 / iterations);  // 80% span / iterations
            
            dispatch_source_t progressTimer = dispatch_source_create(
                DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_main_queue());
            // Fire every 2 seconds (roughly matches per-step duration)
            dispatch_source_set_timer(progressTimer,
                dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
                2 * NSEC_PER_SEC, 0.5 * NSEC_PER_SEC);
            dispatch_source_set_event_handler(progressTimer, ^{
                timerProgress += progressPerTick;
                if (timerProgress > 90) timerProgress = 90;  // cap before completion
                if (progressCallback) {
                    progressCallback(timerProgress, @"Running diffusion...");
                }
            });
            dispatch_resume(progressTimer);
            
            // Also keep the C++ callback in case a future framework rebuild enables it
            auto diffusionProgressCallback = [progressCallback](int step) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (progressCallback) {
                        int progress = 15 + (step * 80 / 100);
                        progressCallback(progress, @"Running diffusion...");
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
                NSLog(@"SanaDiffusionSession: llmOutput shape: %@, size: %zu, order: %d, type: %d",
                      dimStr,
                      llmOutputInfo->size,
                      (int)llmOutputInfo->order,
                      (int)llmOutputInfo->type.code);
            } else {
                NSLog(@"SanaDiffusionSession: llmOutput info is NULL!");
            }
            
            // Use new unified diffusion interface
            bool success = self->mDiffusion->run(
                llmOutput,
                "img2img",                       // mode: image editing
                [normalizedInputPath UTF8String], // input image path
                [outputPath UTF8String],         // output image path
                512,                             // width
                512,                             // height
                iterations,                      // iterNum
                seed,                            // randomSeed
                false,                           // use_cfg
                4.5f,                            // cfg_scale
                diffusionProgressCallback
            );
            
            // Cancel the progress timer now that run() has returned
            dispatch_source_cancel(progressTimer);
            
            
            NSTimeInterval diffusionDuration = [[NSDate date] timeIntervalSinceDate:diffusionStartTime] * 1000;
            NSTimeInterval totalDuration = [[NSDate date] timeIntervalSinceDate:totalStartTime] * 1000;
            
            NSLog(@"SanaDiffusionSession: Diffusion time: %.2f ms", diffusionDuration);
            NSLog(@"SanaDiffusionSession: Total time: %.2f ms", totalDuration);
            
            // Clean up temporary normalized image if it was created
            if (![normalizedInputPath isEqualToString:inputImagePath]) {
                [[NSFileManager defaultManager] removeItemAtPath:normalizedInputPath error:nil];
            }

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
                
                // Note: Do NOT call progressCallback(100) here.
                // The completion handler sets the final message directly,
                // so calling progressCallback would cause a brief flash of
                // an inconsistent message before completion replaces it.
                
                if (completion) {
                    if (success) {
                        completion(YES, nil, totalDuration);
                    } else {
                        completion(NO, @"Diffusion processing failed", totalDuration);
                    }
                }
            });
            
        } @catch (NSException *exception) {
            dispatch_async(dispatch_get_main_queue(), ^{
                self->_isProcessing = NO;
                if (completion) {
                    completion(NO, [NSString stringWithFormat:@"Exception: %@", exception.reason], 0);
                }
            });
        }
    });
}

#pragma mark - Image Orientation Normalization

/// Normalizes the image orientation based on EXIF data.
/// @param imagePath Path to the input image.
/// @return Path to the normalized image, or nil if failed/not needed.
- (NSString *)normalizeImageOrientation:(NSString *)imagePath {
    UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
    if (!image) {
        NSLog(@"SanaDiffusionSession: Failed to load image for normalization: %@", imagePath);
        return nil;
    }
    
    // Check if orientation is already Up
    if (image.imageOrientation == UIImageOrientationUp) {
        return imagePath;
    }
    
    NSLog(@"SanaDiffusionSession: Normalizing image orientation from %ld", (long)image.imageOrientation);
    
    // Drawing into a context redraws the image with "Up" orientation
    UIGraphicsBeginImageContextWithOptions(image.size, NO, image.scale);
    [image drawInRect:CGRectMake(0, 0, image.size.width, image.size.height)];
    UIImage *normalizedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    if (!normalizedImage) {
        NSLog(@"SanaDiffusionSession: Failed to normalize image");
        return nil;
    }
    
    // Save normalized image to a temporary path
    NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"normalized_%@", [imagePath lastPathComponent]]];
    NSData *data = UIImageJPEGRepresentation(normalizedImage, 0.9);
    if ([data writeToFile:tempPath atomically:YES]) {
        NSLog(@"SanaDiffusionSession: Normalized image saved to %@", tempPath);
        return tempPath;
    } else {
        NSLog(@"SanaDiffusionSession: Failed to save normalized image");
        return nil;
    }
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
    mSanaLlm.reset();
    mDiffusion.reset();
    NSLog(@"SanaDiffusionSession deallocated");
}

@end
