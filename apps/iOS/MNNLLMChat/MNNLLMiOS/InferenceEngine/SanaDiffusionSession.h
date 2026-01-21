//
//  SanaDiffusionSession.h
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/01/19.
//  Copyright © 2025 MNN. All rights reserved.
//

#ifndef SanaDiffusionSession_h
#define SanaDiffusionSession_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Completion handler called when model loading finishes.
/// @param success YES if model loaded successfully, NO otherwise.
typedef void (^SanaCompletionHandler)(BOOL success);

/// Progress handler called during style transfer processing.
/// @param progress Current progress percentage (0-100).
/// @param stage Description of the current processing stage.
typedef void (^SanaProgressHandler)(int progress, NSString * _Nonnull stage);

/// Completion handler called when style transfer finishes.
/// @param success YES if transfer completed successfully, NO otherwise.
/// @param error Error message if transfer failed, nil on success.
typedef void (^SanaStyleTransferCompletion)(BOOL success, NSString * _Nullable error);

/// A session object that manages Sana Diffusion style transfer operations.
///
/// Use this class to perform LLM-guided image style transfer using the Sana Diffusion model.
/// The session handles model loading, prompt processing, and the diffusion pipeline.
///
/// ## Overview
///
/// The Sana Diffusion pipeline consists of the following stages:
/// 1. LLM processes the text prompt to generate embeddings
/// 2. VAE Encoder encodes the input image to latent space
/// 3. Transformer performs iterative denoising
/// 4. VAE Decoder produces the final styled image
///
/// ## Usage
///
/// ```objc
/// SanaDiffusionSession *session = [[SanaDiffusionSession alloc]
///     initWithModelPath:@"/path/to/model"
///     completion:^(BOOL success) {
///         if (success) {
///             [session runStyleTransferWithInputImage:@"/path/to/input.jpg"
///                                              prompt:@"Ghibli style"
///                                          outputPath:@"/path/to/output.jpg"
///                                          iterations:20
///                                                seed:-1
///                                    progressCallback:nil
///                                          completion:nil];
///         }
///     }];
/// ```
@interface SanaDiffusionSession : NSObject

/// Initializes a new session with the specified model path.
///
/// Model loading is performed asynchronously on a background thread.
/// The completion handler is called on the main thread when loading finishes.
///
/// @param modelPath Path to the Sana model directory containing:
///                  - `llm/` subdirectory with LLM model files
///                  - `connector.mnn`, `projector.mnn`, `transformer.mnn`
///                  - `vae_encoder.mnn`, `vae_decoder.mnn`
/// @param completion Handler called when model loading completes.
/// @return A new session instance.
- (instancetype)initWithModelPath:(NSString *)modelPath
                       completion:(SanaCompletionHandler _Nullable)completion;

/// Performs style transfer on an input image.
///
/// This method processes the input image using the Sana Diffusion pipeline
/// and saves the styled result to the specified output path.
///
/// @param inputImagePath Path to the input image file.
/// @param prompt The style description prompt (e.g., "Convert to Ghibli style").
/// @param outputPath Destination path for the output image.
/// @param iterations Number of diffusion iterations. Recommended range: 5-20.
/// @param seed Random seed for reproducibility. Use -1 for random seed.
/// @param progressCallback Handler called with progress updates during processing.
/// @param completion Handler called when style transfer completes.
- (void)runStyleTransferWithInputImage:(NSString *)inputImagePath
                                prompt:(NSString *)prompt
                            outputPath:(NSString *)outputPath
                            iterations:(int)iterations
                                  seed:(int)seed
                      progressCallback:(SanaProgressHandler _Nullable)progressCallback
                            completion:(SanaStyleTransferCompletion _Nullable)completion;

/// A Boolean value indicating whether the model is loaded and ready for use.
@property (nonatomic, readonly) BOOL isModelLoaded;

/// A Boolean value indicating whether a style transfer operation is in progress.
@property (nonatomic, readonly) BOOL isProcessing;

/// Returns the default Ghibli style prompt.
/// @return The default Ghibli style transfer prompt string.
+ (NSString *)defaultGhibliPrompt;

/// Returns the file path where benchmark results are saved.
/// @return The absolute path to the benchmark JSON file in the Documents directory.
+ (NSString *)benchmarkFilePath;

@end

NS_ASSUME_NONNULL_END

#endif /* SanaDiffusionSession_h */
