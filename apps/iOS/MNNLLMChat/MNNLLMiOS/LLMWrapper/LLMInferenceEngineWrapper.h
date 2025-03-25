//
//  LLMInferenceEngineWrapper.h
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

#ifndef LLMInferenceEngineWrapper_h
#define LLMInferenceEngineWrapper_h


// LLMInferenceEngineWrapper.h
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^CompletionHandler)(BOOL success);
typedef void (^OutputHandler)(NSString * _Nonnull output);

@interface LLMInferenceEngineWrapper : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath completion:(CompletionHandler)completion;
- (void)processInput:(NSString *)input withOutput:(OutputHandler)output;

- (void)addPromptsFromArray:(NSArray<NSDictionary *> *)array;

- (void)setConfigWithJSONString:(NSString *)jsonStr;

@end

NS_ASSUME_NONNULL_END

#endif /* LLMInferenceEngineWrapper_h */
