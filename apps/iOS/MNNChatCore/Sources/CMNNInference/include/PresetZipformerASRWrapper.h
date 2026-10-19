//
//  PresetZipformerASRWrapper.h
//  MNNLLMiOS
//
//  Thread-owned sherpa-mnn Zipformer wrapper for prerecorded preset audio.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface PresetZipformerASRWrapper : NSObject

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory;

@property(nonatomic, readonly, getter=isReady) BOOL ready;
@property(nonatomic, readonly) NSTimeInterval initializationSeconds;
@property(nonatomic, copy, readonly, nullable) NSString *initializationError;

/// Recognize mono Float32 PCM normalized to [-1, 1]. The native recognizer,
/// stream, and MNN modules live exclusively on one permanent worker thread.
- (nullable NSString *)recognizeFloatPCM:(NSData *)pcm
                              sampleRate:(int32_t)sampleRate
                        inferenceSeconds:(NSTimeInterval *)inferenceSeconds
                                   error:(NSError **)error;

/// Invalidates queued/running output. Native inference may finish in the
/// background, but a canceled result is never returned to the caller.
- (void)cancel;

/// Drains the worker and destroys the recognizer on its owning thread.
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END
