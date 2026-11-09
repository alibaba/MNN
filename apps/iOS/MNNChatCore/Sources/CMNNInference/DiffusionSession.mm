//
//  DiffusionSession.mm
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/22.
//

#import "DiffusionSession.h"

#import <Foundation/Foundation.h>

#include <MNN/diffusion/diffusion.hpp>

using namespace MNN::DIFFUSION;

@implementation DiffusionSession {
    std::shared_ptr<Diffusion> diffusion;
}

- (instancetype)initWithModelPath:(NSString *)modelPath completion:(CompletionHandler)completion {
    self = [super init];
    if (self) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            BOOL success = [self loadModelFromPath:modelPath];
            // MARK: Local Debug
            // BOOL success = [self loadLocalModel];
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(success);
            });
        });
    }
    return self;
}

- (BOOL)loadModelFromPath:(NSString *)modelPath {
    if (!diffusion) {
        Diffusion* rawDiffusion = Diffusion::createDiffusion(
                                        [modelPath UTF8String],
                                     DiffusionModelType::STABLE_DIFFUSION_1_5,
                                     MNNForwardType::MNN_FORWARD_METAL,
                                     2);

        if (rawDiffusion) {
            diffusion = std::shared_ptr<Diffusion>(rawDiffusion);
            return diffusion->load();
        } else {
            return NO;
        }
    }
    return YES;
}


- (BOOL)loadLocalModel {
    if (!diffusion) {
        NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
        Diffusion* rawDiffusion = Diffusion::createDiffusion(
                                        [bundleDirectory UTF8String],
                                        DiffusionModelType::STABLE_DIFFUSION_1_5,
                                        MNNForwardType::MNN_FORWARD_METAL,
                                        2);

        if (rawDiffusion) {
            diffusion = std::shared_ptr<Diffusion>(rawDiffusion);
            return diffusion->load();
        } else {
            return NO;
        }
    }
    return YES;
}

- (void)runWithPrompt:(NSString *)prompt imagePath:(NSString *)imagePath iterations:(int)iterations seed:(int)seed progressCallback:(void (^)(int))progressCallback {
    if (diffusion) {
        NSLog(@"Diffusion model run with iterations: %d, seed: %d", iterations, seed);
        if(diffusion->load()) {
            diffusion->run([prompt UTF8String], [imagePath UTF8String], iterations, seed, progressCallback);
        }
    } else {
        NSLog(@"Diffusion model is not loaded.");
    }
}

@end
