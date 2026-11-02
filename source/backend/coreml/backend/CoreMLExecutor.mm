//
//  CoreMLExecutor.mm
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "CoreMLExecutor.h"
#import <objc/runtime.h>
#include "CoreMLExecutorWrapper.h"
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <mutex>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include "core/TensorUtils.hpp"

bool isAvailable() {
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
        return true;
    }
    return false;
}

namespace {
// Core ML may specialize a compiled bundle on load or on its first prediction.
// Serialize sessions sharing a compiled program.
struct ProgramCacheLock {
    explicit ProgramCacheLock(const char* path) : fd(open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600)) {}
    ProgramCacheLock(const char* path, const char* directory, int owner)
        : fd(open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600)), ownerFd(owner), directory(directory) {}
    ~ProgramCacheLock() {
        if (fd >= 0) {
            close(fd);
        }
        if (ownerFd >= 0) {
            @autoreleasepool {
                NSError* error = nil;
                if (![NSFileManager.defaultManager removeItemAtPath:@(directory.c_str()) error:&error] &&
                    [NSFileManager.defaultManager fileExistsAtPath:@(directory.c_str())]) {
                    NSLog(@"[CoreML] Cannot remove temporary program: %@", error);
                }
            }
            close(ownerFd);
        }
    }
    int fd;
    int ownerFd = -1;
    std::string directory;
    std::mutex mutex;
};
class ProgramCacheGuard {
public:
    explicit ProgramCacheGuard(ProgramCacheLock& lock) : mLock(lock), mThreadLock(lock.mutex) {
        if (lock.fd < 0) {
            return;
        }
        int status;
        do {
            status = flock(lock.fd, LOCK_EX);
        } while (status < 0 && errno == EINTR);
        mLocked = status == 0;
    }
    ~ProgramCacheGuard() {
        if (mLocked) {
            flock(mLock.fd, LOCK_UN);
        }
    }
    bool valid() const { return mLocked; }

private:
    ProgramCacheLock& mLock;
    std::unique_lock<std::mutex> mThreadLock;
    bool mLocked = false;
};

NSString* programRoot() {
    const char* directory = std::getenv("MNN_COREML_CACHE_DIR");
    if (directory && *directory) {
        return [NSString stringWithUTF8String:directory];
    }
    return [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject
        stringByAppendingPathComponent:@"MNN/CoreMLPrograms"];
}

// A locked owner file protects live programs while later processes remove abandoned directories.
std::shared_ptr<ProgramCacheLock> createProgramDirectory(NSString* root) {
    NSFileManager* files = NSFileManager.defaultManager;
    NSError* error = nil;
    if (![files createDirectoryAtPath:root withIntermediateDirectories:YES attributes:nil error:&error]) {
        NSLog(@"[CoreML] Cannot create program directory: %@", error);
        return nullptr;
    }
    NSString* rootLockPath = [root stringByAppendingPathComponent:@".programs.lock"];
    ProgramCacheLock rootLock(rootLockPath.fileSystemRepresentation);
    ProgramCacheGuard guard(rootLock);
    if (!guard.valid()) {
        NSLog(@"[CoreML] Cannot lock program directory: %s", strerror(errno));
        return nullptr;
    }
    NSArray<NSString*>* entries = [files contentsOfDirectoryAtPath:root error:&error];
    if (!entries) {
        NSLog(@"[CoreML] Cannot inspect program directory: %@", error);
        return nullptr;
    }
    for (NSString* entry in entries) {
        if (![entry hasPrefix:@"program-"] || ![[NSUUID alloc] initWithUUIDString:[entry substringFromIndex:8]]) {
            continue;
        }
        NSString* directory = [root stringByAppendingPathComponent:entry];
        NSString* ownerPath = [directory stringByAppendingPathComponent:@"owner.lock"];
        int owner = open(ownerPath.fileSystemRepresentation, O_RDWR | O_CLOEXEC);
        if (owner < 0) {
            if (errno == ENOENT) {
                [files removeItemAtPath:directory error:nil];
            }
            continue;
        }
        if (flock(owner, LOCK_EX | LOCK_NB) == 0) {
            [files removeItemAtPath:directory error:nil];
        }
        close(owner);
    }
    NSString* directory = [root stringByAppendingPathComponent:
        [@"program-" stringByAppendingString:NSUUID.UUID.UUIDString]];
    if (![files createDirectoryAtPath:directory withIntermediateDirectories:NO attributes:nil error:&error]) {
        NSLog(@"[CoreML] Cannot create temporary program: %@", error);
        return nullptr;
    }
    NSString* ownerPath = [directory stringByAppendingPathComponent:@"owner.lock"];
    int owner = open(ownerPath.fileSystemRepresentation, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (owner < 0 || flock(owner, LOCK_EX | LOCK_NB) != 0) {
        if (owner >= 0) {
            close(owner);
        }
        [files removeItemAtPath:directory error:nil];
        return nullptr;
    }
    NSString* predictionLockPath = [directory stringByAppendingPathComponent:@"prediction.lock"];
    return std::make_shared<ProgramCacheLock>(predictionLockPath.fileSystemRepresentation,
                                               directory.fileSystemRepresentation, owner);
}

MLMultiArrayDataType arrayType(const MNN::Tensor* tensor) {
    if (tensor->getType() == halide_type_of<int>()) {
        return MLMultiArrayDataTypeInt32;
    }
    return MLMultiArrayDataTypeFloat16;
}
MLMultiArray* wrapTensor(const MNN::Tensor* tensor, bool flatIO, NSError** error) {
    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];
    auto dimensions = flatIO ? std::vector<int>{static_cast<int>(MNN::TensorUtils::getRawSize(tensor))}
                             : tensor->shape();
    if (dimensions.empty()) {
        dimensions.push_back(1);
    }
    size_t stride = 1;
    for (auto it = dimensions.rbegin(); it != dimensions.rend(); ++it) {
        [shape insertObject:@(*it) atIndex:0];
        [strides insertObject:@(stride) atIndex:0];
        stride *= *it;
    }
    return [[MLMultiArray alloc] initWithDataPointer:tensor->host<void>() shape:shape dataType:arrayType(tensor)
        strides:strides deallocator:^(void*) {} error:error];
}
bool sameStateLayout(MLModel* first, MLModel* second) {
    auto a = first.modelDescription.stateDescriptionsByName;
    auto b = second.modelDescription.stateDescriptionsByName;
    if (a.count != b.count) {
        return false;
    }
    for (NSString* name in a) {
        auto x = a[name].stateConstraint;
        auto y = b[name].stateConstraint;
        if (!y || x.dataType != y.dataType || ![x.bufferShape isEqualToArray:y.bufferShape]) {
            return false;
        }
    }
    return true;
}
bool requiresFileCompilation(const CoreML__Specification__MILSpec__Block* block) {
    for (size_t i = 0; i < block->n_operations; ++i) {
        const auto* op = block->operations[i];
        if (std::strcmp(op->type, "resample") == 0 || std::strcmp(op->type, "topk") == 0) {
            return true;
        }
        for (size_t j = 0; j < op->n_blocks; ++j) {
            if (requiresFileCompilation(op->blocks[j])) {
                return true;
            }
        }
    }
    return false;
}
bool requiresFileCompilation(const CoreML__Specification__Model* spec) {
    const auto* program = spec->mlprogram;
    if (!program) {
        return false;
    }
    for (size_t i = 0; i < program->n_functions; ++i) {
        const auto* function = program->functions[i]->value;
        for (size_t j = 0; j < function->n_block_specializations; ++j) {
            if (requiresFileCompilation(function->block_specializations[j]->value)) {
                return true;
            }
        }
    }
    return false;
}
bool copyOutput(MLMultiArray* array, const MNN::Tensor* tensor) {
    if (!array || array.count != MNN::TensorUtils::getRawSize(tensor) || array.dataType != arrayType(tensor)) {
        return false;
    }
    size_t bytes = tensor->getType().code == halide_type_float ? 2 : tensor->getType().bytes();
    auto src = static_cast<const uint8_t*>(array.dataPointer);
    auto dst = tensor->host<uint8_t>();
    if (src == dst) {
        return true;
    }
    size_t expected = 1;
    bool contiguous = true;
    for (NSInteger i = array.shape.count - 1; i >= 0; --i) {
        if (array.shape[i].integerValue > 1 && array.strides[i].unsignedLongLongValue != expected) {
            contiguous = false;
        }
        expected *= array.shape[i].unsignedLongLongValue;
    }
    if (contiguous) {
        ::memcpy(dst, src, array.count * bytes);
    } else {
        std::vector<size_t> index(array.shape.count, 0);
        for (size_t i = 0; i < array.count; ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < index.size(); ++d) {
                offset += index[d] * array.strides[d].unsignedLongLongValue;
            }
            ::memcpy(dst + i * bytes, src + offset * bytes, bytes);
            for (NSInteger d = index.size() - 1; d >= 0; --d) {
                if (++index[d] < array.shape[d].unsignedLongLongValue) {
                    break;
                }
                index[d] = 0;
            }
        }
    }
    return true;
}
} // namespace

namespace MNN {
std::shared_ptr<void> createCoreMLTemporaryDirectory(std::string& path) {
    @autoreleasepool {
        auto program = createProgramDirectory(programRoot());
        if (!program || program->fd < 0) {
            return nullptr;
        }
        path = program->directory;
        return program;
    }
}
} // namespace MNN

@implementation CoreMLExecutor {
    std::shared_ptr<ProgramCacheLock> _cacheLock;
    std::shared_ptr<std::vector<uint8_t>> _weights;
}
- (void)dealloc {
    self.state = nil;
    self.model = nil;
    _cacheLock.reset();
}
- (bool)buildModel:(CoreML__Specification__Model*)spec weightsPath:(const std::string&)weightsPath
           weights:(const std::shared_ptr<std::vector<uint8_t>>&)weights forceFileCompilation:(bool)forceFileCompilation {
    if (!isAvailable()) {
        return NO;
    }
    // The in-memory asset path changes resample and topk results on some Core ML versions.
    if (forceFileCompilation || requiresFileCompilation(spec)) {
        if (std::getenv("MNN_COREML_VERBOSE")) {
            NSLog(@"[CoreML] Using file compilation for per-command execution or operator compatibility");
        }
        if (!weights) {
            return [self buildModelFromFile:spec weightsPath:weightsPath];
        }
        std::string directory;
        auto cleanup = MNN::createCoreMLTemporaryDirectory(directory);
        if (!cleanup) {
            return NO;
        }
        auto path = directory + "/weight.bin";
        NSData* blob = [NSData dataWithBytesNoCopy:weights->data() length:weights->size() freeWhenDone:NO];
        NSError* error = nil;
        if (![blob writeToFile:@(path.c_str()) options:NSDataWritingAtomic error:&error]) {
            NSLog(@"[CoreML] Cannot write compatibility weights: %@", error);
            return NO;
        }
        return [self buildModelFromFile:spec weightsPath:path];
    }
    self.compiledFromSource = NO;
    size_t size = core_ml__specification__model__get_packed_size(spec);
    NSMutableData* data = [NSMutableData dataWithLength:size];
    core_ml__specification__model__pack(spec, static_cast<uint8_t*>(data.mutableBytes));
    _cacheLock = createProgramDirectory(programRoot());
    if (!_cacheLock) {
        return NO;
    }
    ProgramCacheGuard guard(*_cacheLock);
    if (!guard.valid()) {
        NSLog(@"[CoreML] Cannot lock program cache: %s", strerror(errno));
        return NO;
    }
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    const char* units = std::getenv("MNN_COREML_COMPUTE_UNITS");
    if (units && std::strcmp(units, "cpu") == 0) {
        config.computeUnits = MLComputeUnitsCPUOnly;
    }
    if (units && std::strcmp(units, "all") == 0) {
        config.computeUnits = MLComputeUnitsAll;
    }
    NSError* error = nil;
    NSData* weightData = nil;
    if (weights) {
        _weights = weights;
        weightData = [NSData dataWithBytesNoCopy:_weights->data() length:_weights->size() freeWhenDone:NO];
    } else if (!weightsPath.empty()) {
        weightData = [NSData dataWithContentsOfFile:@(weightsPath.c_str()) options:NSDataReadingMappedIfSafe error:&error];
        if (!weightData) {
            NSLog(@"[CoreML] Cannot read weights: %@", error);
            return NO;
        }
    }
    MLModelAsset* asset = nil;
    if (weightData) {
        NSURL* weightURL = [NSURL fileURLWithPath:@"weights/weight.bin" relativeToURL:nil];
        asset = [MLModelAsset modelAssetWithSpecificationData:data blobMapping:@{weightURL: weightData} error:&error];
    } else {
        asset = [MLModelAsset modelAssetWithSpecificationData:data error:&error];
    }
    if (!asset) {
        NSLog(@"[CoreML] Cannot create model asset: %@", error);
        return NO;
    }
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block MLModel* model = nil;
    __block NSError* loadError = nil;
    [MLModel loadModelAsset:asset configuration:config completionHandler:^(MLModel* loaded, NSError* failure) {
        model = loaded;
        loadError = failure;
        dispatch_semaphore_signal(done);
    }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (!model) {
        NSLog(@"[CoreML] Cannot load model asset: %@", loadError);
        return NO;
    }
    // Core ML may reference the specification and blobs after asynchronous loading completes.
    static char retentionKey;
    NSArray* retained = weightData ? @[asset, data, weightData] : @[asset, data];
    objc_setAssociatedObject(model, &retentionKey, retained, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    self.model = model;
    self.compiledFromSource = YES;
    return YES;
}
- (bool)buildModelFromFile:(CoreML__Specification__Model*)spec weightsPath:(const std::string&)weightsPath {
    if (!isAvailable()) {
        return NO;
    }
    self.compiledFromSource = NO;
    size_t size = core_ml__specification__model__get_packed_size(spec);
    NSMutableData* data = [NSMutableData dataWithLength:size];
    core_ml__specification__model__pack(spec, static_cast<uint8_t*>(data.mutableBytes));
    _cacheLock = createProgramDirectory(programRoot());
    if (!_cacheLock) {
        return NO;
    }
    NSString* directory = @(_cacheLock->directory.c_str());
    NSString* destination = [directory stringByAppendingPathComponent:@"compiled.mlmodelc"];
    NSURL* destinationURL = [NSURL fileURLWithPath:destination isDirectory:YES];
    NSFileManager* files = NSFileManager.defaultManager;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    const char* units = std::getenv("MNN_COREML_COMPUTE_UNITS");
    if (units && std::strcmp(units, "cpu") == 0) {
        config.computeUnits = MLComputeUnitsCPUOnly;
    }
    if (units && std::strcmp(units, "all") == 0) {
        config.computeUnits = MLComputeUnitsAll;
    }
    NSError* error = nil;
    ProgramCacheGuard guard(*_cacheLock);
    if (!guard.valid()) {
        NSLog(@"[CoreML] Cannot lock program cache: %s", strerror(errno));
        return NO;
    }
    NSString* source = [directory stringByAppendingPathComponent:
        weightsPath.empty() ? @"source.mlmodel" : @"source.mlpackage"];
    if (weightsPath.empty()) {
        if (![data writeToFile:source options:NSDataWritingAtomic error:&error]) {
            return NO;
        }
    } else {
        NSString* modelDirectory = [source stringByAppendingPathComponent:@"Data/com.apple.CoreML"];
        NSString* weightsDirectory = [modelDirectory stringByAppendingPathComponent:@"weights"];
        NSString* rootID = NSUUID.UUID.UUIDString;
        NSDictionary* manifest = @{
            @"fileFormatVersion": @"1.0.0", @"rootModelIdentifier": rootID,
            @"itemInfoEntries": @{
                rootID: @{@"author": @"com.apple.CoreML", @"description": @"CoreML Model Specification",
                          @"name": @"model.mlmodel", @"path": @"com.apple.CoreML/model.mlmodel"},
                NSUUID.UUID.UUIDString: @{@"author": @"com.apple.CoreML", @"description": @"CoreML Model Weights",
                          @"name": @"weights", @"path": @"com.apple.CoreML/weights"}}};
        NSData* manifestData = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:&error];
        bool success = manifestData &&
            [files createDirectoryAtPath:weightsDirectory withIntermediateDirectories:YES attributes:nil error:&error] &&
            [data writeToFile:[modelDirectory stringByAppendingPathComponent:@"model.mlmodel"] options:0 error:&error] &&
            [files copyItemAtPath:@(weightsPath.c_str())
                toPath:[weightsDirectory stringByAppendingPathComponent:@"weight.bin"] error:&error] &&
            [manifestData writeToFile:[source stringByAppendingPathComponent:@"Manifest.json"] options:0 error:&error];
        if (!success) {
            [files removeItemAtPath:source error:nil];
            NSLog(@"[CoreML] Cannot prepare weighted program: %@", error);
            return NO;
        }
    }
    NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:source] error:&error];
    [files removeItemAtPath:source error:nil];
    if (!compiled) {
        NSLog(@"[CoreML] Cannot compile kernel: %@", error);
        return NO;
    }
    self.compiledFromSource = YES;
    if (![files moveItemAtURL:compiled toURL:destinationURL error:&error]) {
        [files removeItemAtURL:compiled error:nil];
        NSLog(@"[CoreML] Cannot retain compiled program: %@", error);
        return NO;
    }
    self.model = [MLModel modelWithContentsOfURL:destinationURL configuration:config error:&error];
    if (!self.model) {
        NSLog(@"[CoreML] Cannot load kernel: %@", error);
    }
    return self.model != nil;
}
- (CoreMLExecutor*)sessionWithPrevious:(CoreMLExecutor*)previous {
    CoreMLExecutor* session = [[CoreMLExecutor alloc] init];
    session.model = self.model;
    session->_cacheLock = _cacheLock;
    session->_weights = _weights;
    session.precision = self.precision;
    if (self.model.modelDescription.stateDescriptionsByName.count) {
        if (previous.state && self.model == previous.model) {
            session.state = previous.state;
        } else {
            session.state = [self.model newState];
        }
        if (!session.state) {
            NSLog(@"[CoreML] Cannot create model state");
            return nil;
        }
        if (previous.state && self.model != previous.model) {
            if (!sameStateLayout(self.model, previous.model)) {
                NSLog(@"[CoreML] Cannot reuse KV state with a different buffer layout");
                return nil;
            }
            for (NSString* name in self.model.modelDescription.stateDescriptionsByName) {
                [previous.state getMultiArrayForStateNamed:name handler:^(MLMultiArray* source) {
                    [session.state getMultiArrayForStateNamed:name handler:^(MLMultiArray* destination) {
                        [source transferToMultiArray:destination];
                    }];
                }];
            }
        }
    }
    return session;
}
- (bool)invokeWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)inputs
                 outputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)outputs flatIO:(bool)flatIO {
    @autoreleasepool {
        if (!_cacheLock) {
            return NO;
        }
        ProgramCacheGuard guard(*_cacheLock);
        if (!guard.valid()) {
            NSLog(@"[CoreML] Cannot lock program for prediction: %s", strerror(errno));
            return NO;
        }
        NSError* error = nil;
        NSMutableDictionary<NSString*, MLFeatureValue*>* features = [NSMutableDictionary dictionary];
        for (const auto& input : inputs) {
            auto array = wrapTensor(input.first, flatIO, &error);
            if (!array) {
                NSLog(@"[CoreML] Invalid input %s: %@", input.second.c_str(), error);
                return NO;
            }
            features[@(input.second.c_str())] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        auto provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
        MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
        NSMutableDictionary<NSString*, MLMultiArray*>* backings = [NSMutableDictionary dictionary];
        for (const auto& output : outputs) {
            auto array = wrapTensor(output.first, flatIO, &error);
            if (!array) {
                return NO;
            }
            backings[@(output.second.c_str())] = array;
        }
        options.outputBackings = backings;
        id<MLFeatureProvider> result = nil;
        if (self.state) {
            result = [self.model predictionFromFeatures:provider usingState:self.state options:options error:&error];
        } else {
            result = [self.model predictionFromFeatures:provider options:options error:&error];
        }
        if (!result) {
            NSLog(@"[CoreML] Prediction failed: %@", error);
            return NO;
        }
        for (const auto& output : outputs) {
            if (!copyOutput([result featureValueForName:@(output.second.c_str())].multiArrayValue, output.first)) {
                NSLog(@"[CoreML] Invalid output %s", output.second.c_str());
                return NO;
            }
        }
        return YES;
    }
}
@end
