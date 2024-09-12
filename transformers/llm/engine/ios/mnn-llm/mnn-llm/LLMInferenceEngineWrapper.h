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

typedef void(^ModelLoadingCompletionHandler)(BOOL success);
typedef void (^StreamOutputHandler)(NSString * _Nonnull output);

@interface LLMInferenceEngineWrapper : NSObject

- (instancetype)initWithCompletionHandler:(ModelLoadingCompletionHandler)completionHandler;
- (void)processInput:(NSString *)input withStreamHandler:(StreamOutputHandler)handler;

@end

NS_ASSUME_NONNULL_END

#endif /* LLMInferenceEngineWrapper_h */
