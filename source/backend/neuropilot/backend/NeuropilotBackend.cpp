#include "NeuropilotBackend.hpp"
#ifdef MNN_NEUROPILOT_CONVERT_MODE
#include "converter/ConvertExecution.hpp"
#include "converter/OptimizeCommandBuffer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "MNN/MNNDefine.h"
#include <fstream>
#include <iostream>
#endif
#ifndef MNN_NEUROPILOT_CONVERT_MODE
#ifdef MNN_WITH_PLUGIN
#include "flatbuffers/flexbuffers.h"
#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "shape/SizeComputer.hpp"
#include "common/file_source.h"
#include "executor/neuron_usdk_executor.h"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/OpCommonUtils.hpp"

#endif
#endif
namespace MNN {
#ifdef MNN_NEUROPILOT_CONVERT_MODE
struct TensorInfo {
    std::unique_ptr<tflite::TensorT> tensor;
    std::unique_ptr<tflite::BufferT> buffer;
    bool isInput = false;
    bool isOutput = false;
};
void NeuropilotBackend::prepareTensorQuantInfo(const Tensor* tensor, std::unique_ptr<tflite::QuantizationParametersT>&& param) {
    mQuantInfo.insert(std::make_pair(tensor, std::move(param)));
}
void NeuropilotBackend::setPackTensor(const Tensor* tensor, int packBits) {
    mPackInfo.insert(std::make_pair(tensor, packBits));
}
void NeuropilotBackend::setTensorName(const Tensor* tensor, std::string name) {
    mUserTensorName.insert(std::make_pair(tensor, name));
}
void NeuropilotBackend::insertExtraInput(Tensor* tensor) {
    mExtraInputs.insert(std::make_pair(tensor, mExtraInputs.size()));
}
    
void NeuropilotBackend::insertExtraOutput(Tensor* tensor) {
    mExtraOutputs.insert(std::make_pair(tensor, mExtraOutputs.size()));
}
int NeuropilotBackend::_createTensorFromMNNTensor(const Tensor* tensor, tflite::SubGraphT* dstGraph, std::vector<std::unique_ptr<tflite::BufferT>>& dstBuffers) {
    if (mTensorIndexMap.find(tensor) != mTensorIndexMap.end()) {
        return mTensorIndexMap[tensor];
    }
    TensorInfo info;
    info.tensor.reset(new tflite::TensorT);
    info.buffer.reset(new tflite::BufferT);
    auto& tfliteTensor = info.tensor;
    int tensorIndex = (int)mTensorIndexMap.size();

    mTensorIndexMap[tensor] = tensorIndex;
    auto des = TensorUtils::getDescribe(tensor);
    do {
        if (mStateMask.get() == tensor) {
            tfliteTensor->name = "mask";
            break;
        }
        if (mExtraInputs.find(tensor) != mExtraInputs.end()) {
            tfliteTensor->name = "ei" + std::to_string(mExtraInputs.find(tensor)->second);
            break;
        }
        if (mExtraOutputs.find(tensor) != mExtraOutputs.end()) {
            tfliteTensor->name = "eo" + std::to_string(mExtraOutputs.find(tensor)->second);
            break;
        }
        if (mUserTensorName.find(tensor) == mUserTensorName.end()) {
            if (des->index >= 0) {
                tfliteTensor->name = "t" + std::to_string(des->index);
            } else {
                tfliteTensor->name = "tensor_" + std::to_string(tensorIndex);
            }
        } else {
            tfliteTensor->name = mUserTensorName[tensor];
        }
        if (des->usage == Tensor::InsideDescribe::Usage::INPUT) {
            info.isInput = true;
        }
        if (des->usage == Tensor::InsideDescribe::Usage::OUTPUT) {
            info.isOutput = true;
            if (des->applyQuant) {
                mDequantTensor.insert(std::make_pair(tensorIndex, tensor));
            }
        }
    } while (false);
    tfliteTensor->type = ConvertTflite::getType(tensor);
    if (des->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            MNN_ERROR("Don't support NC4HW4 constant now\n");
        }
        auto size = tensor->usize();
        if (mPackInfo.find(tensor) != mPackInfo.end()) {
            tfliteTensor->type = tflite::TensorType_INT4;
            auto buffer_size = size / 2;
            info.buffer->data.resize(buffer_size);
            auto src_buffer = tensor->host<int8_t>();
            // Ref from kernels/test_util.h
            // Funciton: PackInt4ValuesDenselyInPlace
            for (int i = 0; i < size; ++i) {
                int srcValue = src_buffer[i];
                if (i % 2 == 0) {
                    info.buffer->data[i / 2] = srcValue & 0x0F;
                } else {
                    info.buffer->data[i / 2] |= srcValue << 4;
                }
            }
        } else {
            info.buffer->data.resize(size);
            ::memcpy(info.buffer->data.data(), tensor->host<void>(), size);
        }
    }
    if (mQuantInfo.find(tensor) != mQuantInfo.end()) {
        tfliteTensor->quantization = std::move(mQuantInfo.find(tensor)->second);
        mQuantInfo.erase(tensor);
    } else if (des->applyQuant && des->quantAttr.get() != nullptr) {
        // Load quant info
        auto scale = des->quantAttr->scale;
        auto zero = des->quantAttr->zero;
        tfliteTensor->quantization.reset(new tflite::QuantizationParametersT);
        tfliteTensor->quantization->scale = {scale};
        tfliteTensor->quantization->zero_point = {(long)zero};
        tfliteTensor->quantization->max = {(des->quantAttr->max-zero) * scale};
        tfliteTensor->quantization->min = {(des->quantAttr->min-zero) * scale};
    }

    tfliteTensor->shape = ConvertTflite::getShapeOfTensor(tensor);
    tfliteTensor->buffer = (int)dstBuffers.size();
    dstBuffers.emplace_back(std::move(info.buffer));
    int dstTensorIndex = (int)dstGraph->tensors.size();
    dstGraph->tensors.emplace_back(std::move(info.tensor));
    if (info.isInput) {
        dstGraph->inputs.emplace_back(des->index);
        if (des->index < 0) {
            MNN_ERROR("Invalid Input Tensor for construct Tflite\n");
        }
        mIOIndexMap.insert(std::make_pair(des->index, dstTensorIndex));
    }
    if (info.isOutput) {
        if (des->index < 0) {
            MNN_ERROR("Invalid Output Tensor for construct Tflite\n");
        }
        dstGraph->outputs.emplace_back(des->index);
        mIOIndexMap.insert(std::make_pair(des->index, dstTensorIndex));
    }
    return tensorIndex;
}


Backend* NeuropilotRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new NeuropilotBackend(this);
}

Execution* NeuropilotBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    return new ConvertExecution(this, op);
}
void NeuropilotBackend::onResizeBegin() {
    mInfos.clear();
    mTensorIndexMap.clear();
    mIOIndexMap.clear();
    mPackInfo.clear();
    mDequantTensor.clear();
    mExtraInputs.clear();
    mExtraOutputs.clear();
    mStateMask = nullptr;
    mSharedConst.clear();
}
Tensor* NeuropilotBackend::getConstTensor(std::string name, std::shared_ptr<Tensor> ref) {
    if (mSharedConst.find(name) == mSharedConst.end()) {
        if (ref == nullptr) {
            return nullptr;
        }
        mSharedConst.insert(std::make_pair(name, ref));
        return ref.get();
    }
    return mSharedConst.find(name)->second.get();
}

Tensor* NeuropilotBackend::getStateMask(int maxLength) {
    if (nullptr == mStateMask) {
        mStateMask.reset(Tensor::createDevice<float>({maxLength}));
    }
    return mStateMask.get();
}

ErrorCode NeuropilotBackend::onResizeEnd() {
    // 如果没有操作信息，直接返回
    if (mInfos.empty()) {
        MNN_PRINT("NeuropilotBackend::onResizeEnd: No operations to convert\n");
        return NO_ERROR;
    }
    
    MNN_PRINT("NeuropilotBackend::onResizeEnd: Converting %zu operations to TensorFlow Lite model\n", mInfos.size());
    
    // 创建 TensorFlow Lite 模型
    std::unique_ptr<tflite::ModelT> tfliteModel = createTensorFlowLiteModel();
    if (tfliteModel) {
        MNN_PRINT("Successfully created TensorFlow Lite model with %zu subgraphs and %zu operator codes\n", 
                 tfliteModel->subgraphs.size(), tfliteModel->operator_codes.size());
        
        // 保存模型到缓存路径（如果有的话）
        if (mRuntime && !mRuntime->pCachePath.empty()) {
            std::string modelPath = mRuntime->pCachePath;
            saveTensorFlowLiteModel(tfliteModel, modelPath);
            MNN_PRINT("TensorFlow Lite model conversion completed successfully\n");
        } else {
            MNN_PRINT("TensorFlow Lite model created but no cache path specified for saving\n");
        }
    } else {
        MNN_ERROR("Failed to create TensorFlow Lite model\n");
        return COMPUTE_SIZE_ERROR;
    }
    return NO_ERROR;
}
Backend::MemObj* NeuropilotBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    return new Backend::MemObj;
}

class NeuropilotRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return new NeuropilotRuntime(info);
    }

    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
    static bool _supportQuant(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        auto otype = op->type();
        switch (otype) {
            case OpType_Convolution:
            case OpType_ConvolutionDepthwise:
            {
                if (inputs.size() > 1) {
                    return false;
                }
                auto conv2d = op->main_as_Convolution2D();
                if (nullptr != conv2d && nullptr != conv2d->quanParameter() && conv2d->quanParameter()->type() == 1) {
                    return true;
                }
                return false;
            }
            case OpType_Pooling:
            case OpType_Reshape:
            case OpType_Transpose:
            case OpType_ConvertTensor:
            case OpType_Flatten:
            case OpType_Squeeze:
            case OpType_Unsqueeze:
            case OpType_StridedSlice:
            case OpType_Identity:
            {
                auto input = inputs[0];
                if (input->getType().code != halide_type_float || TensorUtils::getDescribe(input)->quantAttr.get() != TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
                    return false;
                }
            }
                return true;
            default:
                break;
        }
        return false;
    }
    virtual bool onSetQuantInfo(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) const override {
        if (nullptr == op) {
            return true;
        }
        auto res = _supportQuant(op, inputs, outputs);
        for (auto t : outputs) {
            TensorUtils::getDescribe(t)->applyQuant = res;
        }
        return res;
    }
};
// TensorFlow Lite 转换相关的辅助函数实现
std::unique_ptr<tflite::ModelT> NeuropilotBackend::createTensorFlowLiteModel() {
    std::unique_ptr<tflite::ModelT> model(new tflite::ModelT);
    model->version = 3; // TensorFlow Lite 版本
    model->description = "Converted from MNN model";
    
    std::unique_ptr<tflite::SubGraphT> subgraph(new tflite::SubGraphT);
    subgraph->name = "main";
    
    mTensorIndexMap.clear();
    int tensorIndex = 0;
    
    std::vector<std::unique_ptr<tflite::OperatorT>> operators;
    ConvertTflite converter;
    converter.pBackend = this;
    {
        std::unique_ptr<tflite::BufferT> buf(new tflite::BufferT);
        model->buffers.emplace_back(std::move(buf));
    }
    
    ConvertTflite::CommandBuffer totalCmdBuffer;
    for (const auto& info : mInfos) {
        auto cmdbuffer = converter.convert(info.op, info.inputs, info.outputs);
        totalCmdBuffer.extraConst.insert(totalCmdBuffer.extraConst.end(), cmdbuffer.extraConst.begin(), cmdbuffer.extraConst.end());
        for (auto&& cmd : cmdbuffer.commands) {
            totalCmdBuffer.commands.emplace_back(std::move(cmd));
        }
    }
    for (auto& t : totalCmdBuffer.extraConst) {
        _createTensorFromMNNTensor(t.get(), subgraph.get(), model->buffers);
    }
    OptimizeCommandBuffer opt(&converter);
    totalCmdBuffer = opt.reduce(std::move(totalCmdBuffer));
    for (auto& cmd : totalCmdBuffer.commands) {
        if (cmd.op.get() == nullptr) {
            continue;
        }
        for (const auto& inputTensor : cmd.inputs) {
            _createTensorFromMNNTensor(inputTensor, subgraph.get(), model->buffers);
        }
        for (const auto& inputTensor : cmd.outputs) {
            _createTensorFromMNNTensor(inputTensor, subgraph.get(), model->buffers);
        }
        auto op = std::move(cmd.op);
        for (const auto& inputTensor : cmd.inputs) {
            op->inputs.push_back(mTensorIndexMap[inputTensor]);
        }
        for (const auto& outputTensor : cmd.outputs) {
            op->outputs.push_back(mTensorIndexMap[outputTensor]);
        }
        operators.push_back(std::move(op));
    }
    std::vector<std::unique_ptr<tflite::OperatorT>> deqOp;
    std::vector<std::shared_ptr<Tensor>> deqTensors;
    for (auto& iter : mDequantTensor) {
        auto originIndex = iter.first;
        std::shared_ptr<Tensor> floatTensor(new Tensor(iter.second, iter.second->getDimensionType(), false));
        TensorUtils::getDescribe(floatTensor.get())->dimensionFormat = TensorUtils::getDescribe(iter.second)->dimensionFormat;
        deqTensors.emplace_back(floatTensor);
        auto newIndex = _createTensorFromMNNTensor(floatTensor.get(), subgraph.get(), model->buffers);
        // Swap tensor info
        auto swapT = std::move(subgraph->tensors[newIndex]);
        subgraph->tensors[newIndex] = std::move(subgraph->tensors[originIndex]);
        subgraph->tensors[originIndex] = std::move(swapT);
        // Swap name
        auto name = subgraph->tensors[newIndex]->name;
        subgraph->tensors[newIndex]->name = subgraph->tensors[originIndex]->name;
        subgraph->tensors[originIndex]->name = name;
        // Reset op output index
        for (auto& op : operators) {
            for (int i=0; i<op->inputs.size(); ++i) {
                if (op->inputs[i] == originIndex) {
                    op->inputs[i] = newIndex;
                }
            }
            for (int i=0; i<op->outputs.size(); ++i) {
                if (op->outputs[i] == originIndex) {
                    op->outputs[i] = newIndex;
                }
            }
        }
        // Add Dequant Op
        std::unique_ptr<tflite::OperatorT> dequantOp(new tflite::OperatorT);
        dequantOp->opcode_index = converter.getOpIndex(tflite::BuiltinOperator_DEQUANTIZE);
        dequantOp->inputs = {newIndex};
        dequantOp->outputs = {originIndex};
        deqOp.emplace_back(std::move(dequantOp));
    }
    for (auto&& op : deqOp) {
        operators.emplace_back(std::move(op));
    }
    subgraph->operators = std::move(operators);
    // Reindex subgraph io
    std::sort(subgraph->inputs.begin(), subgraph->inputs.end());
    std::sort(subgraph->outputs.begin(), subgraph->outputs.end());
    for (int i=0; i<subgraph->inputs.size(); ++i) {
        subgraph->inputs[i] = mIOIndexMap[subgraph->inputs[i]];
    }
    for (int i=0; i<subgraph->outputs.size(); ++i) {
        subgraph->outputs[i] = mIOIndexMap[subgraph->outputs[i]];
    }
    // Insert Extra Input and Output
    if (nullptr != mStateMask) {
        auto index = mTensorIndexMap.find(mStateMask.get())->second;
        subgraph->inputs.emplace_back(index);
    }
    for (int i=0; i<mExtraInputs.size(); ++i) {
        for (auto& iter : mExtraInputs) {
            if (iter.second == i) {
                auto index = mTensorIndexMap.find(iter.first)->second;
                subgraph->inputs.emplace_back(index);
                break;
            }
        }
    }
    for (int i=0; i<mExtraOutputs.size(); ++i) {
        for (auto& iter : mExtraOutputs) {
            if (iter.second == i) {
                auto index = mTensorIndexMap.find(iter.first)->second;
                subgraph->outputs.emplace_back(index);
                break;
            }
        }
    }
    model->subgraphs.push_back(std::move(subgraph));
    model->operator_codes = converter.releaseCodes();
    
    return model;
}

void NeuropilotBackend::saveTensorFlowLiteModel(std::unique_ptr<tflite::ModelT>& model, const std::string& filePath) {
    flatbuffers::FlatBufferBuilder builder;
    builder.ForceDefaults(true);
    auto modelOffset = tflite::Model::Pack(builder, model.get());
    // Must use tflite::FinishModelBuffer, otherwise can't verify success
    tflite::FinishModelBuffer(builder, modelOffset);
    
    std::ofstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        file.close();
        MNN_PRINT("TensorFlow Lite model saved to: %s\n", filePath.c_str());
    } else {
        MNN_ERROR("Failed to save TensorFlow Lite model to: %s\n", filePath.c_str());
    }
}

#endif

#ifndef MNN_NEUROPILOT_CONVERT_MODE
#ifdef MNN_WITH_PLUGIN

namespace plugin {
static bool computeIndex(const std::vector<Tensor *> & inputs, const Attribute* attrAllShape, int & index) {
    std::string inputShapeStr = "";
    for (int i = 0; i < inputs.size(); i++) {
        if (i > 0) {
            inputShapeStr += "_";
        }
        std::vector<int> inputShape = inputs[i]->shape();
        for (int j = 0; j < inputShape.size(); j++) {
            if (j > 0) {
                inputShapeStr += "x";
            }
            inputShapeStr += std::to_string(inputShape[j]);
        }
    }

    std::vector<std::string> vec;
    if (nullptr != attrAllShape && nullptr != attrAllShape->list() && nullptr != attrAllShape->list()->s()) {
        for (int i = 0; i < attrAllShape->list()->s()->size(); ++i) {
            vec.push_back(attrAllShape->list()->s()->GetAsString(i)->str());
        }
    } else {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op.\n");
        return false;
    }

    auto it = std::find(vec.begin(), vec.end(), inputShapeStr);
    if (it != vec.end()) {
        index = std::distance(vec.begin(), it);
        return true;
    }
    return false;
}

static std::vector<std::string> _divide(std::string content, const std::string& divide) {
    auto pos = content.find(divide);
    if (pos == std::string::npos) {
        return {content};
    }
    std::vector<std::string> res;
    int sta = 0;
    do {
        res.emplace_back(content.substr(sta, pos - sta));
        sta = pos + divide.size();
        pos = content.find(divide, sta);
    } while (pos != std::string::npos);
    res.emplace_back(content.substr(sta, std::string::npos));
    return res;
}
static std::vector<std::vector<int>> _extractShapes(std::string shapeString) {
    // Divide by "_"
    auto inputs = _divide(shapeString, "_");
    std::vector<std::vector<int>> res;
    for (auto s : inputs) {
        auto dims = _divide(s, "x");
        std::vector<int> dim;
        for (auto t : dims) {
            std::istringstream _os(t);
            int dimU;
            _os >> dimU;
            dim.emplace_back(dimU);
        }
        res.emplace_back(dim);
    }
    return res;
}
namespace shape_inference {
class PluginShapeRaw : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};


bool PluginShapeRaw::compute(InferShapeContext* ctx) {
    if (ctx->hasAttr("op")) {
        auto attr = ctx->getAttr("op");
        if (nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto realop = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            return SizeComputer::computeOutputSize(realop, ctx->inputs(), ctx->outputs());
        }
    } else {
        int shapeIndex = 0;
        auto attrAllShape = ctx->getAttr("allInputShape");

        if (!(computeIndex(ctx->inputs(), attrAllShape, shapeIndex))) {
            MNN_ERROR("Failed to compute shape for Plugin Op.\n");
            return false;
        }

        std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
        for (int i=0; i<ctx->outputs().size(); ++i) {
            auto dst = ctx->output(i);
            std::string key = prefix + std::to_string(i);
            auto attr = ctx->getAttr(key.c_str());
            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_QNN: Failed to find raw shape %s.\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j=0; j<blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
    return false;
}
}
namespace backend {
class PluginExecuteRaw : public CPUComputeKernel {
private:
    std::vector<std::pair<const MNN::Tensor *, size_t>> mInputs;
    std::vector<std::pair<const MNN::Tensor *, size_t>> mOutputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealInputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealOutputs;
    std::shared_ptr<mtk::NeuronUsdkExecutor> mExecutor;

    std::unique_ptr<mtk::SharedWeightsHandle> mSharedWeightsHandle;
    std::vector<std::shared_ptr<mtk::NeuronUsdkExecutor>> mAllExecutors;
    std::string mPath;
    struct StateTensor {
        std::vector<float> data;
        int inside;
        int outside;
    };
    std::vector<float> mMask;
    std::vector<StateTensor> mStateInput;
    int mStateCurrent = 0;
    int mStateMaxSize = 0;
public:
    ~ PluginExecuteRaw() {
        mRealInputs.clear();
        mRealOutputs.clear();
        mExecutor.reset();
    }
    bool init(CPUKernelContext* ctx) override {
        auto state = ctx->getAttr("state");
        int stateNumber = 0;
        if (nullptr != state) {
            int axis = 0;
            auto ref = flexbuffers::GetRoot(state->tensor()->uint8s()->data(), state->tensor()->uint8s()->size());
            auto refMap = ref.AsMap();
            auto keys = refMap.Keys();
            std::vector<std::vector<int>> stateShape;
            for (int i=0; i<keys.size(); ++i) {
                auto key = keys[i].AsKey();
                if (std::string(key) == "number") {
                    stateNumber = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "max_length") {
                    mStateMaxSize = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "axis") {
                    axis = refMap.Values()[i].AsInt32();
                    continue;
                }
                if (std::string(key) == "shape") {
                    auto shapeVectors = refMap.Values()[i].AsVector();
                    for (int u=0; u<shapeVectors.size(); ++u) {
                        auto shapeV = shapeVectors[u].AsVector();
                        std::vector<int> shapes;
                        for (int v=0; v<shapeV.size(); ++v) {
                            shapes.emplace_back(shapeV[v].AsInt32());
                        }
                        stateShape.emplace_back(shapes);
                    }
                    continue;
                }
            }
            mMask = std::vector<float>(mStateMaxSize, -20000.0f);
            mStateInput.resize(stateShape.size());
            for (int i=0; i<stateShape.size(); ++i) {
                auto& shape = stateShape[i];
                auto& input = mStateInput[i];
                input.outside = 1;
                for (int j=0; j<axis; ++j) {
                    input.outside *= shape[j];
                }
                auto axisLength = shape[axis];
                MNN_ASSERT(1 == axisLength);
                input.inside = 1;
                for (int j=axis+1; j<shape.size(); ++j) {
                    input.inside *= shape[j];
                }
                input.data = std::vector<float>(input.outside*input.inside*axisLength*mStateMaxSize, 0.0f);
            }
        }
        FUNC_PRINT(stateNumber);
        int maskNumber = stateNumber > 0 ? 1 : 0;
        auto allInputShape = ctx->getAttr("allInputShape");
        auto inputTensor = ctx->inputs();
        auto outputTensor = ctx->outputs();
        mPath = ctx->getAttr("path")->s()->str();
        if (nullptr != allInputShape->list() && allInputShape->list()->s()->size() > 1) {
            // ShareWeight
            {
                auto path = ctx->dir_path() + mPath + ".weight";
                FileSource files(path.c_str());
                mSharedWeightsHandle.reset(new mtk::SharedWeightsHandle({files}, 1));
                mSharedWeightsHandle->preload();
            }
            auto ShareWeight = mSharedWeightsHandle->getSharedWeights(0);
            // AllExecutor
            mAllExecutors.resize(allInputShape->list()->s()->size());
            for (int i = 0; i < allInputShape->list()->s()->size(); ++i) {
                auto shape = _extractShapes(allInputShape->list()->s()->GetAsString(i)->str());
                auto path = ctx->dir_path()  + mPath + ".shared_" + std::to_string(i);
                FileSource files(path.c_str());
                // Input + mask + state
                int inputSize = inputTensor.size() + maskNumber + stateNumber;
                for (auto& s : shape) {
                    int size = 1;
                    for (auto d : s) {
                        size *= d;
                    }
                    if (1 == size) {
                        inputSize -= 1;
                    }
                }
                mAllExecutors[i].reset(new mtk::NeuronUsdkExecutor(inputSize, files, ShareWeight));
                // Add shared weight input
                mAllExecutors[i]->setNumInputs(inputSize+1);
                mAllExecutors[i]->setNumOutputs(outputTensor.size()+stateNumber);
                mAllExecutors[i]->initialize();
            }
            return true;
        }
        auto path = ctx->dir_path() + mPath;
        FileSource files(path.c_str());
        mExecutor.reset(new mtk::NeuronUsdkExecutor(inputTensor.size(), files));
        mExecutor->setNumInputs(inputTensor.size() + maskNumber + stateNumber);
        mExecutor->setNumOutputs(outputTensor.size() + stateNumber);
        mExecutor->initialize();
        return true;
    }
    bool resize(CPUKernelContext* ctx) override {
        if (!mAllExecutors.empty()) {
            int index = -1;
            auto res = computeIndex(ctx->inputs(), ctx->getAttr("allInputShape"), index);
            if (!res) {
                return false;
            }
            mExecutor = mAllExecutors[index];
        }
        auto inputTensor = ctx->inputs();
        mInputs.resize(inputTensor.size());
        mRealInputs.resize(inputTensor.size());
        mInputs.clear();
        mRealInputs.clear();
        for (int i=0; i<inputTensor.size(); ++i) {
            // For Neuropilot will remove all scalar inputs offline, because it may cause crash
            if (inputTensor[i]->elementSize() == 1) {
                continue;
            }
            std::shared_ptr<MNN::Tensor> realInput;
            std::pair<const MNN::Tensor *, size_t> inputPair;
            if (TensorUtils::getDescribe(inputTensor[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                realInput.reset(new Tensor(inputTensor[i], Tensor::TENSORFLOW));
                inputPair.first = mRealInputs[i].get();
                inputPair.second = mRealInputs[i]->usize();
            } else {
                inputPair.first = inputTensor[i];
                inputPair.second = inputTensor[i]->usize();
            }
            mInputs.emplace_back(inputPair);
            mRealInputs.emplace_back(realInput);
        }
        auto outputTensor = ctx->outputs();
        mOutputs.resize(outputTensor.size());
        mRealOutputs.resize(outputTensor.size());
        for (int i=0; i<outputTensor.size(); ++i) {
            if (TensorUtils::getDescribe(outputTensor[i])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
                mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::TENSORFLOW));
                mOutputs[i].first = mRealOutputs[i].get();
            } else {
                mOutputs[i].first = outputTensor[i];
            }
            mOutputs[i].second = mOutputs[i].first->usize();
        }
        return true;
    }
    bool compute(CPUKernelContext* ctx) override {
        auto inputTensor = ctx->inputs();
        for (int i=0; i<mInputs.size(); ++i) {
            if (mRealInputs[i] != nullptr) {
                MNNCPUCopyBuffer(inputTensor[i], mRealInputs[i].get());
            }
            auto& buffer = mExecutor->getInput(i);
            if (mInputs[i].second == buffer.sizeBytes) {
                ::memcpy(buffer.buffer, mInputs[i].first->host<void>(), buffer.sizeBytes);
            } else if (mInputs[i].second == buffer.sizeBytes * 2) {
                // Float2Half
                MNNQuantizeFP16(mInputs[i].first->host<float>(), (int16_t*)buffer.buffer, buffer.sizeBytes / 2);
            } else {
                MNN_ERROR("For %s, %d input size not math: needed: %ld: input: %ld\n", mPath.c_str(), i, mExecutor->getModelInputSizeBytes(), mInputs[i].second);
            }
        }
        // If has remove, remove invalid state
        auto meta = (KVMeta*)(ctx->backend()->getMetaPtr());
        if (nullptr != meta && mStateInput.size() > 0) {
            if (meta->remove > 0) {
                mStateCurrent-= meta->remove;
                // TODO: If meta->remove larger then meta->add, need clean stateinput
                for (int i=0; i<meta->remove; ++i) {
                    mMask[i+mStateCurrent] = -20000.0f;
                }
            }
        }
        // Set State Mask
        if (!mStateInput.empty()) {
            auto& buffer = mExecutor->getInput(mInputs.size());
            ::memcpy(buffer.buffer, mMask.data(), buffer.sizeBytes);
        }
        // Copy State
        for (int i=0; i<mStateInput.size(); ++i) {
            auto& input = mStateInput[i];
            auto& buffer = mExecutor->getInput(i + 1 + mInputs.size());
            ::memcpy(buffer.buffer, input.data.data(), buffer.sizeBytes);
        }
        mExecutor->runInference();
        // Update State
        if (nullptr != meta && mStateInput.size() > 0) {
            for (int i=0; i<meta->add; ++i) {
                mMask[i+mStateCurrent] = 0.0f;
            }
            // Temply use StateOutputs[0] size to compute seq_len
            int seqLen = mExecutor->getModelOutputSizeBytes(mOutputs.size()) / mStateInput[0].inside / sizeof(float) / mStateInput[0].outside;
            for (int i=0; i<mStateInput.size(); ++i) {
                auto& input = mStateInput[i];
                auto& buffer = mExecutor->getOutput(i + mOutputs.size());
                for (int y=0; y<input.outside; ++y) {
                    auto dstOffset = y * input.inside * mStateMaxSize + mStateCurrent * input.inside;
                    auto srcOffset = y * input.inside * seqLen;
                    auto dst = input.data.data() + dstOffset;
                    auto src = (float*)buffer.buffer + srcOffset;
                    ::memcpy(dst, src, meta->add * input.inside * sizeof(float));
                }
            }
            mStateCurrent += meta->add;
        }

        auto outputTensor = ctx->outputs();
        for (int i=0; i<mOutputs.size(); ++i) {
            auto& buffer = mExecutor->getOutput(i);
            if (mOutputs[i].second == buffer.sizeBytes) {
                ::memcpy(mOutputs[i].first->buffer().host, buffer.buffer, mOutputs[i].first->usize());
            } else if (mOutputs[i].second == buffer.sizeBytes * 2) {
                MNNDequantizeFP16((int16_t*)buffer.buffer, mOutputs[i].first->host<float>(), buffer.sizeBytes / 2);
            } else {
                MNN_ERROR("For %s, %d output size not math: needed: %ld: real: %ld\n", mPath.c_str(), i, buffer.sizeBytes, mOutputs[i].second);
            }
            if (mRealOutputs[i] != nullptr) {
                MNNCPUCopyBuffer(mRealOutputs[i].get(), outputTensor[i]);
            }
        }
        return true;
    }
};
} // namespace backend

}
#endif
#endif
void registerNeuroPilot() {
#ifdef MNN_NEUROPILOT_CONVERT_MODE
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new NeuropilotRuntimeCreator, false);
#else
#ifdef MNN_WITH_PLUGIN
    plugin::InferShapeKernelRegister::add("Neuropilot", []() { // NOLINT
        return new plugin::shape_inference::PluginShapeRaw;               // NOLINT
    });
    plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("Neuropilot", []() {
        return new plugin::backend::PluginExecuteRaw;
    });
#endif
#endif
}
}
