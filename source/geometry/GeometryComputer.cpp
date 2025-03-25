//
//  GeometryComputer.cpp
//  MNN
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>
#include <MNN/Interpreter.hpp>
#include "geometry/GeometryComputer.hpp"
#include "core/Backend.hpp"
#include "core/OpCommonUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

GeometryComputer::Context::~Context() {
    // Do nothing
}
GeometryComputer::Context::Context(int mask, std::shared_ptr<Backend> allocBackend, MNNForwardType type, BackendConfig::PrecisionMode precision) : mMask(mask) {
    mBackend       = allocBackend;
    flatbuffers::FlatBufferBuilder builder(32);
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Raster);
    auto lastOffset = opBuilder.Finish();
    builder.Finish(lastOffset);
    mRasterOp.reset(new BufferStorage);
    mRasterOp->storage = builder.ReleaseRaw(mRasterOp->allocated_size, mRasterOp->offset);
    mForwardType = type;
    mPrecision = precision;
}

void GeometryComputer::Context::clear() {
    mTempConstTensors.clear();
}
const std::vector<std::shared_ptr<Tensor>>& GeometryComputer::Context::searchConst(const Op* op) {
    auto iter = mConstTensors.find(op);
    if (iter == mConstTensors.end()) {
        mConstTensors.insert(std::make_pair(op, std::vector<std::shared_ptr<Tensor>>{}));
        return mEmpty;
    }
    return iter->second;
}
std::shared_ptr<Tensor> GeometryComputer::Context::allocConst(const Op* key, const std::vector<int>& shape,
                                                              halide_type_t type, Tensor::DimensionType dimType) {
    std::shared_ptr<Tensor> tensor(Tensor::createDevice(shape, type, dimType));
    TensorUtils::getDescribe(tensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
    auto res                                      = mBackend->onAcquireBuffer(tensor.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    TensorUtils::getDescribeOrigin(tensor.get())->setBackend(mBackend.get());
    auto iter = mConstTensors.find(key);
    if (iter != mConstTensors.end()) {
        iter->second.emplace_back(tensor);
    } else {
        mTempConstTensors.emplace_back(tensor);
    }
    return tensor;
}

bool GeometryComputer::Context::allocTensor(Tensor* tensor) {
    auto res = mBackend->onAcquireBuffer(tensor, Backend::STATIC);
    if (!res) {
        return false;
    }
    TensorUtils::getDescribe(tensor)->usage = Tensor::InsideDescribe::CONSTANT;
    TensorUtils::getDescribeOrigin(tensor)->setBackend(mBackend.get());
    return true;
}

inline bool _hasZeroDim(const Tensor* t) {

    for (int i = 0; i < t->dimensions(); ++i) {
        if (t->length(i) <= 0) {
            return true;
        }
    }
    return false;
}

static bool _virtualMemory(Tensor::InsideDescribe::NativeInsideDescribe* des) {
    return des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL &&  nullptr == des->rasterCommand.lock().get();
}

bool GeometryComputer::ComputePermuteRegion(Tensor* input, Tensor* output, int* newshape, int shapeDim) {
    auto inputDes   = TensorUtils::getDescribe(input);
    auto outputDes  = TensorUtils::getDescribe(output);
    MNN_ASSERT(input->dimensions() >= 1);
    MNN_ASSERT(output->dimensions() == input->dimensions());
    MNN_ASSERT(shapeDim == input->dimensions());
    auto originTensor = input;
    int shape[MNN_MAX_TENSOR_DIM];
    if (nullptr != newshape) {
        for (int i = 0; i < input->buffer().dimensions; ++i) {
            shape[i] = newshape[i];
        }
    } else {
        for (int i = 0; i < input->buffer().dimensions; ++i) {
            shape[i] = input->buffer().dimensions - i - 1;
        }
    }
    
    int inputShape[MNN_MAX_TENSOR_DIM];
    int inputStrides[MNN_MAX_TENSOR_DIM];
    int inputShapeSize = 0;
    int preAxis = -2;
    for (int i=0; i<input->buffer().dimensions; ++i) {
        auto axis = shape[i];
        auto len = input->length(axis);
        if (1 == len) {
            continue;
        }
        if (axis - preAxis == 1) {
            // Fuse dimension if possible
            inputShape[inputShapeSize - 1] *= len;
        } else {
            if (preAxis >= 0) {
                // Compute last stride
                int stride = 1;
                for (int v=preAxis+1; v < input->buffer().dimensions; ++v) {
                    stride *= input->length(v);
                }
                inputStrides[inputShapeSize - 1] = stride;
            }
            inputShapeSize+=1;
            inputShape[inputShapeSize - 1] = len;
        }
        preAxis = shape[i];
    }
    if (preAxis >= 0) {
        // Compute last stride
        int stride = 1;
        for (int v=preAxis+1; v < input->buffer().dimensions; ++v) {
            stride *= input->length(v);
        }
        inputStrides[inputShapeSize - 1] = stride;
    }
    if (0 == inputShapeSize) {
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions = {TensorUtils::makeFullSlice(input)};
        return true;
    }
    int outputStrides[MNN_MAX_TENSOR_DIM];
    {
        int stride = 1;
        for (int i=inputShapeSize-1; i>=0; --i) {
            outputStrides[i] = stride;
            stride *= inputShape[i];
        }
    }
    /** Move max three inputShapeSize to last three location.
     * Don't change max three number relative position
     * */
bool isReorderShape = false;
isReorderShape = (inputShapeSize > 4);
    if (inputShapeSize == 4) {
        // TODO: Opt this logic
    isReorderShape = (inputShape[0] > inputShape[1] + inputShape[2] + inputShape[3]);
    }
    if (isReorderShape) {
        int max1 = inputShape[0], max2 = -1, max3 = -1;
        // Find Max Three Number
        for (int i = 1; i < inputShapeSize; i++) {
            if (inputShape[i] > max1) {
                max3 = max2;
                max2 = max1;
                max1 = inputShape[i];
            } else if (inputShape[i] > max2) {
                max3 = max2;
                max2 = inputShape[i];
            }
            else if (inputShape[i] > max3) {
                max3 = inputShape[i];
            }
        }
        
        // Move Max Three Number to Last Location
        int lastIndex = inputShapeSize-1;
        for (int i = inputShapeSize-1; i >= 0; i--) {
            if (inputShape[i] == max1) {
                if(i != lastIndex) {
                    std::swap(inputShape[i], inputShape[lastIndex]);
                    std::swap(inputStrides[i], inputStrides[lastIndex]);
                    std::swap(outputStrides[i], outputStrides[lastIndex]);
                }
                max1 = -1;
                lastIndex--;
            } else if (inputShape[i] == max2) {
                if(i != lastIndex) {
                    std::swap(inputShape[i], inputShape[lastIndex]);
                    std::swap(inputStrides[i], inputStrides[lastIndex]);
                    std::swap(outputStrides[i], outputStrides[lastIndex]);
                }
                max2 = -1;
                lastIndex--;
            } else if (inputShape[i] == max3) {
                if(i != lastIndex) {
                    std::swap(inputShape[i], inputShape[lastIndex]);
                    std::swap(inputStrides[i], inputStrides[lastIndex]);
                    std::swap(outputStrides[i], outputStrides[lastIndex]);
                }
                max3 = -1;
                lastIndex--;
            }
            if(lastIndex < inputShapeSize-3) {
                break;
            }
        }
    }
// Compute inside, outside, axis
    int inside        = 1;
    int insideStride  = 0;
    int outside       = 1;
    int outsideStride = 0;
    int axis          = 1;
    int axisStride    = 0;
    int breakAxis     = -1;
    int remainSize    = 1;
    int outputInsideStride = 0;
    int outputAxisStride = 0;
    int outputOutsideStride = 0;
    {
        if (inputShapeSize >= 1) {
            inside       = inputShape[inputShapeSize-1];
            insideStride = inputStrides[inputShapeSize-1];
            outputInsideStride = outputStrides[inputShapeSize-1];
        }
        if (inputShapeSize >= 2) {
            axis       = inputShape[inputShapeSize-2];
            axisStride = inputStrides[inputShapeSize-2];
            outputAxisStride = outputStrides[inputShapeSize-2];
        }
        if (inputShapeSize >= 3) {
            outside       = inputShape[inputShapeSize-3];
            outsideStride = inputStrides[inputShapeSize-3];
            outputOutsideStride = outputStrides[inputShapeSize-3];
            breakAxis     = inputShapeSize - 3;
            for (int i = 0; i < inputShapeSize - 3; ++i) {
                remainSize *= inputShape[i];
            }
        }
    }
    outputDes->regions.resize(remainSize);
    outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    int32_t mod[MNN_MAX_TENSOR_DIM];
    for (int i = 0; i < breakAxis; ++i) {
        int value = 1;
        for (int j = i + 1; j < breakAxis; ++j) {
            value *= inputShape[j];
        }
        mod[i] = value;
    }
    for (int indice = 0; indice < remainSize; ++indice) {
        int value       = indice;
        int inputOffset = 0;
        int outputOffset = 0;
        for (int i = 0; i < breakAxis; ++i) {
            auto coordinate = value / mod[i];
            inputOffset += coordinate * inputStrides[i];
            outputOffset += coordinate * outputStrides[i];
            value = value % mod[i];
        }
        Tensor::InsideDescribe::Region& slice = outputDes->regions[indice];
        slice.src.offset                      = inputOffset;
        slice.src.stride[0]                   = outsideStride;
        slice.size[0]                         = outside;
        slice.src.stride[1]                   = axisStride;
        slice.size[1]                         = axis;
        slice.src.stride[2]                   = insideStride;
        slice.size[2]                         = inside;
        slice.origin                          = originTensor;
        slice.dst.offset                      = outputOffset;
        slice.dst.stride[0]                   = outputOutsideStride;
        slice.dst.stride[1]                   = outputAxisStride;
        slice.dst.stride[2]                   = outputInsideStride;
    }
    return true;
}

void GeometryComputer::Context::getRasterCacheCreateRecursive(Tensor* src, CommandBuffer& cmd) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (!_virtualMemory(srcDes)) {
        return;
    }
    if (_hasZeroDim(src)) {
        return;
    }
    bool needDelete = false;
    bool supportFuse = support(Interpreter::GEOMETRCOMPUTEMASK_FUSEREGION);
    bool supportFuseMulti = support(Interpreter::GEOMETRCOMPUTEMASK_FUSEREGION_MULTI);
    for (int regIndex = 0; regIndex < srcDes->regions.size();) {
        auto input = srcDes->regions.data() + regIndex;
        MNN_ASSERT(input->origin != src);
        
        auto inputDes = TensorUtils::getDescribe(input->origin);
        while (_virtualMemory(inputDes) && supportFuse) {
            if (0 == inputDes->regions.size()) {
                // Empty Input, Remove the region by set size as 0
                input->size[0] = 0;
                needDelete = true;
                break;
            }
            if (1 < inputDes->regions.size()) {
                if (!supportFuseMulti) {
                    break;
                }
                bool allCanMerge = true;
                for (auto& reg : inputDes->regions) {
                    allCanMerge = allCanMerge && mFuseUtils.match(reg, *input);
                    if (!allCanMerge) {
                        break;
                    }
                }
                if (!allCanMerge) {
                    break;
                }
                Tensor::InsideDescribe::Region backup = *input;
                mFuseUtils.match(inputDes->regions[0], *input);
                mFuseUtils.apply(inputDes->regions[0], *input);
                for (int i=1; i<inputDes->regions.size(); ++i) {
                    auto newReg = backup;
                    mFuseUtils.match(inputDes->regions[i], newReg);
                    mFuseUtils.apply(inputDes->regions[i], newReg);
                    if (newReg.size[0] == 0) {
                        continue;
                    }
                    srcDes->regions.emplace_back(newReg);
                }
                // After emplace_back, the input will change, reref it
                input = srcDes->regions.data() + regIndex;
                if (input->size[0] == 0) {
                    needDelete = true;
                    break;
                }
                inputDes = TensorUtils::getDescribe(input->origin);
                continue;
            }
            bool merge = mFuseUtils.match(inputDes->regions[0], *input);
            if (merge) {
                mFuseUtils.apply(inputDes->regions[0], *input);
            } else {
                break;
            }
            if (input->size[0] == 0) {
                needDelete = true;
                break;
            }
            inputDes = TensorUtils::getDescribe(input->origin);
        }
        if (input->size[0] > 0) {
            getRasterCacheCreateRecursive(input->origin, cmd);
        }
        ++regIndex;
    }
    if (needDelete) {
        auto regions = std::move(srcDes->regions);
        srcDes->regions.reserve(regions.size());
        for (int regIndex = 0; regIndex < regions.size(); ++regIndex) {
            auto input = std::move(regions[regIndex]);
            if (input.size[0] == 0 || input.size[1] == 0 || input.size[2] == 0) {
                continue;
            }
            srcDes->regions.emplace_back(std::move(input));
        }
    }
    getRasterCacheCreate(src, cmd);
}
void GeometryComputer::Context::getRasterCacheCreate(Tensor* src, CommandBuffer& cmdBuffer) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (!_virtualMemory(srcDes)) {
        return;
    }
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.op = flatbuffers::GetRoot<Op>(mRasterOp->buffer());
    cmd.buffer = mRasterOp;
    cmd.outputs = {src};
    TensorUtils::setRasterInputs(cmdP.get());
    srcDes->rasterCommand = std::weak_ptr<Command>(cmdP);
    cmdBuffer.command.emplace_back(std::move(cmdP));
//    srcDes->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
    return;
}

bool DefaultGeometryComputer::onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         Context& context, CommandBuffer& cmd) const {
    if (1 != cmd.command.size()) {
        return false;
    }
    return true;
}

bool DefaultGeometryComputer::onCompute(const Op* op, const std::vector<Tensor*>& originInputs,
                                        const std::vector<Tensor*>& outputs, GeometryComputer::Context& context,
                                        CommandBuffer& res) const {
    auto inputs = originInputs;
    // Last Command
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.op      = op;
    cmd.inputs  = std::move(inputs);
    cmd.outputs = std::move(outputs);
    res.command.emplace_back(std::move(cmdP));
    return true;
}

class GeometryComputerManager {
public:
    GeometryComputer* search(int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Origin == compType) {
            return &mDefault;
        }
        if (Runtime::Compiler_Loop == compType) {
            auto iter = mLoopTable[type].get();
            if (iter != nullptr) {
                return iter;
            }
        }
        // Geometry
        auto iter = mTable[type].get();
        if (iter != nullptr) {
            // FUNC_PRINT(type);
            return iter;
        }
        return &mDefault;
    }
    static void init() {
        gInstance = new GeometryComputerManager;
        gInstance->mTable.resize(OpType_MAX + 1);
        gInstance->mLoopTable.resize(OpType_MAX + 1);
    }
    static GeometryComputerManager* get() {
        return gInstance;
    }
    void insert(std::shared_ptr<GeometryComputer> c, int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Geometry == compType) {
            mTable[type] = c;
        } else if (Runtime::Compiler_Loop == compType) {
            mLoopTable[type] = c;
        }
    }
private:
    std::vector<std::shared_ptr<GeometryComputer>> mTable;
    std::vector<std::shared_ptr<GeometryComputer>> mLoopTable;
    static GeometryComputerManager* gInstance;
    DefaultGeometryComputer mDefault;
};

GeometryComputerManager* GeometryComputerManager::gInstance;
void GeometryComputer::registerGeometryComputer(std::shared_ptr<GeometryComputer> comp, std::vector<int> type, Runtime::CompilerType compType) {
    auto ins = GeometryComputerManager::get();
    for (auto t : type) {
        ins->insert(comp, t, compType);
    }
}
void GeometryComputer::init() {
    if (nullptr == GeometryComputerManager::get()) {
        GeometryComputerManager::init();
        registerGeometryOps();
    }
}

const GeometryComputer* GeometryComputer::search(int type, Runtime::CompilerType compType) {
    return GeometryComputerManager::get()->search(type, compType);
}
} // namespace MNN
