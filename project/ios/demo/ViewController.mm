//
//  ViewController.mm
//  MNN
//
//  Created by MNN on 2019/02/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "ViewController.h"
#import <Metal/Metal.h>

#import <AVFoundation/AVFoundation.h>
#import <MNN/HalideRuntime.h>
#import <MNN/MNNDefine.h>
#import <MNN/ErrorCode.hpp>
#import <MNN/ImageProcess.hpp>
#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#define MNN_METAL
#import <MNN/MNNSharedContext.h>

#include <thread>
typedef struct {
    float value;
    int index;
} LabeledElement;

static int CompareElements(const LabeledElement *a, const LabeledElement *b) {
    if (a->value > b->value) {
        return -1;
    } else if (a->value < b->value) {
        return 1;
    } else {
        return 0;
    }
}

struct PretreatInfo {
    int outputSize[4];
    float mean[4];
    float normal[4];
    float inputSize[4];
    float matrix[16];
};
struct GpuCache {
    CVMetalTextureCacheRef _textureCache;
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _pretreat;
    id<MTLFunction> _function;
    id<MTLBuffer> _constant;
    id<MTLCommandQueue> _queue;
    GpuCache() {
        _device = MTLCreateSystemDefaultDevice();
        CVReturn res = CVMetalTextureCacheCreate(nil, nil, _device, nil, &_textureCache);
        FUNC_PRINT(res);
        id<MTLLibrary> library = [_device newDefaultLibrary];
        _function = [library newFunctionWithName:@"pretreat"];
        NSError* error = nil;
        _pretreat = [_device newComputePipelineStateWithFunction:_function error:&error];
        _constant = [_device newBufferWithLength:sizeof(PretreatInfo) options:MTLCPUCacheModeDefaultCache];
        _queue = [_device newCommandQueue];
    }
    ~ GpuCache() {
        
    }
};
@interface Model : NSObject {
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_session;
    std::mutex _mutex;
    MNNForwardType _type;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<GpuCache> _cache;
}
@property (strong, nonatomic) UIImage *defaultImage;
@property (strong, nonatomic) NSArray<NSString *> *labels;

@end

@implementation Model
- (void)setType:(MNNForwardType)type threads:(NSUInteger)threads {
    std::unique_lock<std::mutex> _l(_mutex);
    if (_session) {
        _net->releaseSession(_session);
    }
    if (nullptr == _cache) {
        _cache.reset(new GpuCache);
    }
    MNN::ScheduleConfig config;
    config.type      = type;
    config.numThread = (int)threads;
    if (type == MNN_FORWARD_METAL) {
        MNN::BackendConfig bnConfig;
        MNNMetalSharedContext context;
        context.device = _cache->_device;
        context.queue = _cache->_queue;
        bnConfig.sharedContext = &context;
        config.backendConfig = &bnConfig;
        _session = _net->createSession(config);
    } else {
        _session = _net->createSession(config);
    }
    _input = _net->getSessionInput(_session, nullptr);
    _output = _net->getSessionOutput(_session, nullptr);
    _type = type;
}
- (NSString *)benchmark:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    if (!_net || !_session) {
        return nil;
    }
    MNN::Tensor *output = _net->getSessionOutput(_session, nullptr);
    MNN::Tensor copy(output);
    auto input = _net->getSessionInput(_session, nullptr);
    MNN::Tensor tensorCache(input);
    input->copyToHostTensor(&tensorCache);

    // run
    NSTimeInterval begin = NSDate.timeIntervalSinceReferenceDate;
    // you should set input data for each inference
    for (int i = 0; i < cycles; i++) {
        input->copyFromHostTensor(&tensorCache);
        _net->runSession(_session);
        output->copyToHostTensor(&copy);
    }
    NSTimeInterval cost = NSDate.timeIntervalSinceReferenceDate - begin;
    NSString *string = @"";
    return [string stringByAppendingFormat:@"time elapse: %.3f ms", cost * 1000.f / cycles];
}


- (NSString *)inferNoLock:(NSInteger)cycles {
    if (!_net || !_session) {
        return nil;
    }
    // run
    NSTimeInterval begin = NSDate.timeIntervalSinceReferenceDate;
    // you should set input data for each inference
    _net->runSession(_session);

    MNN::Tensor *output = _net->getSessionOutput(_session, nullptr);
    MNN::Tensor copy(output);
    output->copyToHostTensor(&copy);
    NSTimeInterval cost = NSDate.timeIntervalSinceReferenceDate - begin;

    // result
    float *data = copy.host<float>();
    LabeledElement objects[1000];
    for (int i = 0; i < 1000; i++) {
        objects[i].value = data[i];
        objects[i].index = i;
    }
    qsort(objects, 1000, sizeof(objects[0]), (int (*)(const void *, const void *))CompareElements);

    // to string
    NSString *string = @"";
    for (int i = 0; i < 3; i++) {
        string = [string stringByAppendingFormat:@"%@: %f\n", _labels[objects[i].index], objects[i].value];
    }
    return [string stringByAppendingFormat:@"time elapse: %.3f ms", cost * 1000.f / 1.0f];
}

- (NSString *)infer:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    return [self inferNoLock:cycles];
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    return [self infer:cycles];
}

- (NSString *)inferBuffer:(CMSampleBufferRef)buffer {
    return [self infer:1];
}
@end

#pragma mark -
@interface MobileNetV2 : Model
@end

@implementation MobileNetV2
- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *labels  = [[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"];
        NSString *lines   = [NSString stringWithContentsOfFile:labels encoding:NSUTF8StringEncoding error:nil];
        self.labels       = [lines componentsSeparatedByString:@"\n"];
        self.defaultImage = [UIImage imageNamed:@"testcat.jpg"];

        NSString *model = [[NSBundle mainBundle] pathForResource:@"mobilenet_v2.caffe" ofType:@"mnn"];
        _net            = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.UTF8String));
    }
    return self;
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    int w               = image.size.width;
    int h               = image.size.height;
    unsigned char *rgba = (unsigned char *)calloc(w * h * 4, sizeof(unsigned char));
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef    = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }

    const float means[3]   = {103.94f, 116.78f, 123.68f};
    const float normals[3] = {0.017f, 0.017f, 0.017f};
    auto pretreat          = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(MNN::CV::RGBA, MNN::CV::BGR, means, 3, normals, 3));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 223.0, (h - 1) / 223.0);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(rgba, w, h, 0, input);
    free(rgba);

    return [super inferNoLock:0];
}

- (NSString *)inferBuffer:(CMSampleBufferRef)sampleBuffer {
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    std::unique_lock<std::mutex> _l(_mutex);

    // GPU
    if (_type == MNN_FORWARD_METAL) {
        size_t width = CVPixelBufferGetWidth(pixelBuffer);
        size_t height = CVPixelBufferGetHeight(pixelBuffer);
        MTLPixelFormat pixelFormat = MTLPixelFormatBGRA8Unorm;

        CVMetalTextureRef texture = NULL;
        CVReturn status = CVMetalTextureCacheCreateTextureFromImage(NULL, _cache->_textureCache, pixelBuffer, NULL, pixelFormat, width, height, 0, &texture);
        id<MTLTexture> inputTexture = CVMetalTextureGetTexture(texture);
        CVBufferRelease(texture);
        PretreatInfo pretreat;
        // TODO: Only copy it once
        pretreat.outputSize[0] = 224;
        pretreat.outputSize[1] = 224;
        pretreat.mean[0] = 103.94f;
        pretreat.mean[1] = 116.78f;
        pretreat.mean[2] = 123.68f;
        pretreat.mean[3] = 0.0f;
        pretreat.normal[0] = 0.017f;
        pretreat.normal[1] = 0.017f;
        pretreat.normal[2] = 0.017f;
        pretreat.normal[3] = 0.0f;
        ::memcpy([_cache->_constant contents], &pretreat, sizeof(PretreatInfo));
        auto cmd = [_cache->_queue commandBuffer];
        auto enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_cache->_pretreat];
        [enc setTexture:inputTexture atIndex:0];
        [enc setBuffer:_cache->_constant offset:0 atIndex:1];
        MNNMetalTensorContent sharedContent;
        _input->getDeviceInfo(&sharedContent, MNN_FORWARD_METAL);
        // For Metal Context to write, don't need finish, just use flush
        _input->wait(MNN::Tensor::MAP_TENSOR_WRITE, false);
        [enc setBuffer:sharedContent.buffer offset:sharedContent.offset atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(28, 28, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];        
        return [super inferNoLock:0];
    }

    // CPU
    int w                        = (int)CVPixelBufferGetWidth(pixelBuffer);
    int h                        = (int)CVPixelBufferGetHeight(pixelBuffer);
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    unsigned char *bgra = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    const float means[3]   = {103.94f, 116.78f, 123.68f};
    const float normals[3] = {0.017f, 0.017f, 0.017f};
    auto pretreat          = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(MNN::CV::BGRA, MNN::CV::BGR, means, 3, normals, 3));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 223.0, (h - 1) / 223.0);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(bgra, w, h, 0, input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return [super inferNoLock:0];
}

@end

#pragma mark -
@interface SqueezeNetV1_1 : Model
@end

@implementation SqueezeNetV1_1

- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *labels  = [[NSBundle mainBundle] pathForResource:@"squeezenet" ofType:@"txt"];
        NSString *lines   = [NSString stringWithContentsOfFile:labels encoding:NSUTF8StringEncoding error:nil];
        self.labels       = [lines componentsSeparatedByString:@"\n"];
        self.defaultImage = [UIImage imageNamed:@"squeezenet.jpg"];

        NSString *model = [[NSBundle mainBundle] pathForResource:@"squeezenet_v1.1.caffe" ofType:@"mnn"];
        _net            = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.UTF8String));
    }
    return self;
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    int w               = image.size.width;
    int h               = image.size.height;
    unsigned char *rgba = (unsigned char *)calloc(w * h * 4, sizeof(unsigned char));
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef    = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }

    const float means[3] = {104.f, 117.f, 123.f};
    MNN::CV::ImageProcess::Config process;
    ::memcpy(process.mean, means, sizeof(means));
    process.sourceFormat = MNN::CV::RGBA;
    process.destFormat   = MNN::CV::BGR;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(process));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 226.f, (h - 1) / 226.f);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(rgba, w, h, 0, input);
    free(rgba);

    return [super inferNoLock:0];
}

- (NSString *)inferBuffer:(CMSampleBufferRef)sampleBuffer {
    std::unique_lock<std::mutex> _l(_mutex);
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    int w                        = (int)CVPixelBufferGetWidth(pixelBuffer);
    int h                        = (int)CVPixelBufferGetHeight(pixelBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    unsigned char *bgra = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    const float means[3] = {104.f, 117.f, 123.f};
    MNN::CV::ImageProcess::Config process;
    ::memcpy(process.mean, means, sizeof(means));
    process.sourceFormat = MNN::CV::BGRA;
    process.destFormat   = MNN::CV::BGR;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(process));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 226.f, (h - 1) / 226.f);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(bgra, w, h, 0, input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return [super inferNoLock:0];
}
@end

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
@property (assign, nonatomic) MNNForwardType forwardType;
@property (assign, nonatomic) int threadCount;

@property (strong, nonatomic) Model *mobileNetV2;
@property (strong, nonatomic) Model *squeezeNetV1_1;
@property (strong, nonatomic) Model *currentModel;

@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *modelItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *forwardItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *threadItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *runItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *benchmarkItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *cameraItem;

@end

@implementation ViewController

- (void)awakeFromNib {
    [super awakeFromNib];

    self.forwardType    = MNN_FORWARD_CPU;
    self.threadCount    = 4;
    self.mobileNetV2    = [MobileNetV2 new];
    self.squeezeNetV1_1 = [SqueezeNetV1_1 new];
    self.currentModel   = self.mobileNetV2;

    AVCaptureSession *session        = [[AVCaptureSession alloc] init];
    session.sessionPreset            = AVCaptureSessionPreset1280x720;
    AVCaptureDevice *device          = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *input      = [[AVCaptureDeviceInput alloc] initWithDevice:device error:NULL];
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [output setSampleBufferDelegate:self queue:dispatch_queue_create("video_infer", 0)];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};

    if ([session canAddInput:input]) {
        [session addInput:input];
    }
    if ([session canAddOutput:output]) {
        [session addOutput:output];
    }
    [session commitConfiguration];

    self.session = session;
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self refresh];
}

- (void)refresh {
    [_currentModel setType:_forwardType threads:_threadCount];
    [self run];
}

- (IBAction)toggleInput {
    if (_session.running) {
        [self usePhotoInput];
        [self run];
    } else {
        [self useCameraInput];
    }
}

- (void)useCameraInput {
    [_session startRunning];
    self.navigationItem.leftBarButtonItem.title = @"Photo";
    self.runItem.enabled                        = NO;
    self.benchmarkItem.enabled                  = NO;
}

- (void)usePhotoInput {
    [_session stopRunning];
    _imageView.image                            = _currentModel.defaultImage;
    self.navigationItem.leftBarButtonItem.title = @"Camera";
    self.runItem.enabled                        = YES;
    self.benchmarkItem.enabled                  = YES;
}

- (IBAction)toggleModel {
    __weak typeof(self) weakify = self;
    UIAlertController *alert    = [UIAlertController alertControllerWithTitle:@"选择模型"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"MobileNet V2"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.modelItem.title          = action.title;
                                                self.currentModel             = self.mobileNetV2;
                                                if (!self.session.running) {
                                                    self.imageView.image = self.currentModel.defaultImage;
                                                }
                                                [self refresh];
                                            }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"SqueezeNet V1.1"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.modelItem.title          = action.title;
                                                self.currentModel             = self.squeezeNetV1_1;
                                                if (!self.session.running) {
                                                    self.imageView.image = self.currentModel.defaultImage;
                                                }
                                                [self refresh];
                                            }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleMode {
    __weak typeof(self) weakify = self;
    UIAlertController *alert    = [UIAlertController alertControllerWithTitle:@"运行模式"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"CPU"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.forwardItem.title        = action.title;
                                                self.forwardType              = MNN_FORWARD_CPU;
                                                [self refresh];
                                            }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"Metal"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.forwardItem.title        = action.title;
                                                self.forwardType              = MNN_FORWARD_METAL;
                                                [self refresh];
                                            }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleThreads {
    __weak typeof(self) weakify       = self;
    void (^onToggle)(UIAlertAction *) = ^(UIAlertAction *action) {
        __strong typeof(weakify) self = weakify;
        self.threadItem.title         = [NSString stringWithFormat:@"%@", action.title];
        self.threadCount              = action.title.intValue;
        [self refresh];
    };
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Thread Count"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"1" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"2" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"4" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"8" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"10" style:UIAlertActionStyleDefault handler:onToggle]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)run {
    if (!_session.running) {
        self.resultLabel.text = [_currentModel inferImage:_imageView.image cycles:1];
    }
}

- (IBAction)benchmark {
    if (!_session.running) {
        self.cameraItem.enabled    = NO;
        self.runItem.enabled       = NO;
        self.benchmarkItem.enabled = NO;
        self.modelItem.enabled     = NO;
        self.forwardItem.enabled   = NO;
        self.threadItem.enabled    = NO;
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSString *str = [self->_currentModel benchmark:100];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.resultLabel.text      = str;
                self.cameraItem.enabled    = YES;
                self.runItem.enabled       = YES;
                self.benchmarkItem.enabled = YES;
                self.modelItem.enabled     = YES;
                self.forwardItem.enabled   = YES;
                self.threadItem.enabled    = YES;
            });
        });
    }
}

#pragma mark AVCaptureAudioDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
    CIImage *ci        = [[CIImage alloc] initWithCVPixelBuffer:CMSampleBufferGetImageBuffer(sampleBuffer)];
    CIContext *context = [[CIContext alloc] init];
    CGImageRef cg      = [context createCGImage:ci fromRect:ci.extent];

    UIImageOrientation orientaion;
    switch (connection.videoOrientation) {
        case AVCaptureVideoOrientationPortrait:
            orientaion = UIImageOrientationUp;
            break;
        case AVCaptureVideoOrientationPortraitUpsideDown:
            orientaion = UIImageOrientationDown;
            break;
        case AVCaptureVideoOrientationLandscapeRight:
            orientaion = UIImageOrientationRight;
            break;
        case AVCaptureVideoOrientationLandscapeLeft:
            orientaion = UIImageOrientationLeft;
            break;
        default:
            break;
    }

    UIImage *image = [UIImage imageWithCGImage:cg scale:1.f orientation:orientaion];
    CGImageRelease(cg);
    NSString *result = [_currentModel inferBuffer:sampleBuffer];

    dispatch_async(dispatch_get_main_queue(), ^{
        if (self.session.running) {
            self.imageView.image  = image;
            self.resultLabel.text = result;
        }
    });
}

@end
