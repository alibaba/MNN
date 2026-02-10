//
//  DiffusionSession.h
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/22.
//

#ifndef DiffusionSession_h
#define DiffusionSession_h

#import <Foundation/Foundation.h>

typedef void (^CompletionHandler)(BOOL success);
typedef void (^OutputHandler)(NSString * _Nonnull output);

@interface DiffusionSession : NSObject

- (instancetype)initWithModelPath:(NSString * _Nonnull)modelPath 
                      completion:(CompletionHandler _Nullable)completion;

- (void)runWithPrompt:(NSString *)prompt 
            imagePath:(NSString *)imagePath 
           iterations:(int)iterations 
                 seed:(int)seed
      progressCallback:(void (^)(int))progressCallback;

@end

#endif /* DiffusionSession_h */
