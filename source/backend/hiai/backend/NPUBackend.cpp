//
//  NPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBackend.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#ifdef HIAI_DEBUG
    #include <android/log.h>
    #include <sys/time.h>
#endif
namespace MNN {

    void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
        for (z = 0; z < depth; ++z) {
            int plane         = z / 4;
            uint8_t* dstPlane = plane * area * 4 + dst;
            int offset        = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(float));
        for (z = 0; z < depth; ++z) {
            int plane       = z / 4;
            float* dstPlane = plane * area * 4 + dst;
            int offset      = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void NHWC2NCHW(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i * c;
                auto dstArea = dstBatch + i;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci * area] = srcArea[ci];
                }
            }
        }
    }

    void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane             = z / 4;
            const float* srcPlane = plane * area * 4 + src;
            int offset            = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane               = z / 4;
            const uint8_t* srcPlane = plane * area * 4 + src;
            int offset              = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void NCHW2NHWC(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i;
                auto dstArea = dstBatch + i * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci] = srcArea[ci * area];
                }
            }
        }
    }

    ErrorCode tensorConvert(const Tensor* input, const Tensor* output) {
        auto ib     = input->buffer();
        auto ob     = output->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
        if (ib.dimensions <= 1 || source == dest) {
            ::memcpy(ob.host, ib.host, input->size());
            return NO_ERROR;
        }
        if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
            MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
            return INVALID_VALUE;
        }
        int area = 1, batch = ib.dim[0].extent, channel;
        if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
            channel = ib.dim[1].extent;
            for (int axis = 2; axis < ib.dimensions; ++axis) {
                area *= ib.dim[axis].extent;
            }
        } else {
            channel = ib.dim[ib.dimensions - 1].extent;
            for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
                area *= ib.dim[axis].extent;
            }
        }
        const int bitLength = ib.type.bytes();

        if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNUnpackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i,
                                    (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

        if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNPackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i, (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

       if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NHWC2NCHW((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NCHW2NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else {
            return NOT_SUPPORT;
        }

        return NO_ERROR;
    }

    static inline std::map<OpType, NPUBackend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, NPUBackend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, NPUBackend::Creator*>; });
        return ret;
    }

    bool NPUBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_PRINT("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    NPUBackend::NPUBackend(const NPURuntime* runtime) : Backend(MNN_FORWARD_USER_0) {
        mNPURuntime = runtime;
        mPrecision  = mNPURuntime->mPrecision;
#ifdef HIAI_DEBUG
        // Retrieve a handle to libandroid.
        void *lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
        // Access the native tracing functions.
        if (lib != NULL) {
            // Use dlsym() to prevent crashes on devices running Android 5.1
            // (API level 22) or lower.
            ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
                dlsym(lib, "ATrace_beginSection"));
            ATrace_endSection = reinterpret_cast<fp_ATrace_endSection>(
                dlsym(lib, "ATrace_endSection"));
            MNN_PRINT("get function ptr :%p,%p",ATrace_beginSection, ATrace_endSection);
        }
#endif
    }
    NPUBackend::~NPUBackend() {

    }

    void NPUBackend::setNetworkInput(const std::vector<Tensor *> &inputs, const Op* op) {
       for (size_t i = 0; i < op->inputIndexes()->size(); i++) {
            auto inputIndex = op->inputIndexes()->data()[i];
            auto outputIndex = op->outputIndexes()->data()[i];
            Tensor *inputTensor = inputs[i];
            bool isInput = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
            if (isInput && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("input") + to_string(inputIndex);
                shared_ptr<hiai::op::Data> data(new hiai::op::Data(opName));
                vector<int64_t> dims;
                for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                    dims.push_back(inputTensor->buffer().dim[i].extent);
                }
                ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                if (TensorUtils::getDescribe(inputTensor)->dimensionFormat == MNN_DATA_FORMAT::MNN_DATA_FORMAT_NHWC) {
                    desc.SetFormat(ge::FORMAT_NHWC);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                    desc.SetDataType(ge::DT_INT32);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                    desc.SetDataType(ge::DT_INT64);
                }
                data->update_input_desc_x(desc);
                // map
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(data, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
                std::pair<int, std::vector<ge::Operator>> item(outputIndex, {*data.get()});
                mInputOps.insert(item);
            }

            bool isConst = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;
            if (isConst && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("Const") + to_string(inputIndex);
                shared_ptr<hiai::op::Const> mConst(new hiai::op::Const(opName));
                {
                    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
                    vector<int64_t> dims;
                    for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                        dims.push_back(inputTensor->buffer().dim[i].extent);
                    }
                    ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        fdesc.SetDataType(ge::DT_INT32);
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        fdesc.SetDataType(ge::DT_INT64);
                    }
                    filter->SetTensorDesc(fdesc);
                    filter->SetData((uint8_t *)inputTensor->host<float>(), inputTensor->elementSize() * sizeof(float));
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        filter->SetData((uint8_t *)inputTensor->host<int32_t>(), inputTensor->elementSize() * sizeof(int32_t));
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        filter->SetData((uint8_t *)inputTensor->host<int64_t>(), inputTensor->elementSize() * sizeof(int64_t));
                    }
                    mConst->set_attr_value(filter);
                }
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(mConst, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
            }
        }
    }

    Execution* NPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        
        if (iter == map->end()) {
            MNN_ERROR("map not find !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return nullptr;
        }

        auto exe = iter->second->onCreate(inputs, outputs, op, this);

        if (nullptr == exe) {
            MNN_ERROR("nullptr == exe !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return nullptr;
        }

        return exe;
    }

    void NPUBackend::NPUBackend::onExecuteBegin() const {
    }
    
    void NPUBackend::onExecuteEnd() const {
        process();
    }

    Backend::MemObj* NPUBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        if(isInputCopy){
            mInputMap.insert(make_pair((unsigned long)tensor, mInputMap.size()));
        }
        // Don't need extra release
        return new Backend::MemObj;
    }

    bool NPUBackend::onClearBuffer() {
        return true;
    }

    void NPUBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef HIAI_DEBUG
        ATrace_beginSection("onCopy");
#endif
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

        if (isConst) {
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            tmpTensor->buffer().host = srcTensor->buffer().host;
            return;
        }
        
        if (isInputCopy) {
            auto index = mInputMap.find((unsigned long)(const_cast<Tensor*>(dstTensor)));
            MNN_ASSERT(index != mInputMap.end());
            shared_ptr<hiai::INDTensorBuffer> input = inputTensors[index->second];
            memcpy(input->GetData(), srcTensor->host<void>(), (size_t)input->GetSize());
        } else if(isOutputCopy){
            int index;
            bool flag = false;
            for(index = 0; index < mMNNOutTensors.size(); index++) {
                if(mMNNOutTensors[index] == srcTensor) {
                    flag = true;
                    break;
                }
            }
            if(flag == false) {
                MNN_PRINT("MNNTensor and HIAITensor mismatch!");
                return;
            }
            shared_ptr<hiai::INDTensorBuffer> output = outputTensors[index];
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            memcpy(tmpTensor->buffer().host, output->GetData(), (size_t)output->GetSize());
        }
#ifdef HIAI_DEBUG
        ATrace_endSection();
#endif
    }

    void NPUBackend::onResizeBegin() {
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        inputTensors.clear();
        outputTensors.clear();
        mMNNOutTensors.clear();
        mSclipMap.clear();
    }

    ErrorCode NPUBackend::onResizeEnd() {
        return bulidIRModelAndLoad();
    }

    ErrorCode NPUBackend::bulidIRModelAndLoad() {
        std::vector<ge::Operator> inputs;
        for (auto input : mInputOps){
            inputs.push_back(input.second[0]);
        }
        std::vector<ge::Operator> outputOps;
        for (auto outOp : mOutGEOpMap) {
            outputOps.push_back(*outOp.first.get());
        }
        MNN_PRINT("mOutputOps : %lu \n", outputOps.size());

        string graphName = string("Graph1");
        string version = string("model_v000011");
        string modelName = to_string(0);
        mModelName.push_back(modelName);
        ge::Graph graph(graphName);
        graph.SetInputs(inputs).SetOutputs(outputOps);

        std::shared_ptr<ge::Model> model = std::make_shared<ge::Model>("model", graphName);
        if (model == nullptr) {
            MNN_ERROR("Create model fail.");
            return INVALID_VALUE;
        }

        model->SetGraph(graph);

        hiai::ModelBuildOptions buildOptions;
        buildOptions.formatMode = hiai::FormatMode::USE_ORIGIN;
        std::ifstream file("quant_param", std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            MNN_PRINT("no quant_param config file, build non-quantized model.\n");
        } else {
            MNN_PRINT("find quant_param config file, build quantized model.\n");
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::string buffer(size, ' ');
            if (!file.read(&buffer[0], size)) {
                MNN_ERROR("Failed to read file.\n");
                return INVALID_VALUE;
            }
            file.close();
            buildOptions.quantizeConfig = buffer;
        }
        domi::HiaiIrBuild modelBuilder;
        auto ret = modelBuilder.Build(buildOptions, modelName, model, builtModel);
        if (ret != hiai::SUCCESS || builtModel == nullptr) {
            MNN_ERROR("model build fail !\n");
            return INVALID_VALUE;
        }
#ifdef HIAI_DEBUG
        ret = builtModel->SaveToFile("/data/local/tmp/test_quant.om");
        if (ret != hiai::SUCCESS) {
            MNN_ERROR("builtModel SaveToFile failed\n");
            return INVALID_VALUE;
        }
#endif
        modelManager = hiai::CreateModelManager();
        hiai::ModelInitOptions initOptions;
        ret = modelManager->Init(initOptions, builtModel, nullptr);
        if (ret != hiai::SUCCESS) {
            MNN_ERROR("modelManager Init failed");
            return INVALID_VALUE;
        }
        ret = modelManager->SetPriority(hiai::ModelPriority::PRIORITY_HIGH);
        if (ret != hiai::SUCCESS) {
            MNN_ERROR("modelManager SetPriority failed");
            return INVALID_VALUE;
        }
        std::vector<hiai::NDTensorDesc> inputDesc = builtModel->GetInputTensorDescs();
        for (size_t i = 0; i < inputDesc.size(); i++) {
            std::shared_ptr<hiai::INDTensorBuffer> inputTensorBuffer = hiai::CreateNDTensorBuffer(inputDesc[i]);
            inputTensors.push_back(inputTensorBuffer);
        }
        std::vector<hiai::NDTensorDesc> outputDesc = builtModel->GetOutputTensorDescs();
        for (size_t i = 0; i < outputDesc.size(); i++) {
            std::shared_ptr<hiai::INDTensorBuffer> outputTensorBuffer = hiai::CreateNDTensorBuffer(outputDesc[i]);
            outputTensors.push_back(outputTensorBuffer);
        }
        auto index = 0;
        for (auto opMap : mOutGEOpMap) {
            for (auto tensor : opMap.second) {
                mMNNOutTensors.push_back(tensor);
                index++;
            }
        }
        return NO_ERROR;
    }

    int NPUBackend::process() const {
        auto ret = modelManager->Run(*(const_cast<vector<shared_ptr<hiai::INDTensorBuffer>>*>(&inputTensors)), *(const_cast<vector<shared_ptr<hiai::INDTensorBuffer>>*>(&outputTensors)));
        return ret;
    }

    shared_ptr<ge::Operator> NPUBackend::getInputOps(const Op *op, int index) {
        vector<shared_ptr<ge::Operator>> ops;
        bool find = false;
        for (size_t i = 0; i < op->inputIndexes()->size(); i++){
            auto inputIndex = op->inputIndexes()->data()[i];
            // printf("inputIndex : %d \n", inputIndex);
            auto iter = mGrapMap.find(inputIndex);
            if(iter != mGrapMap.end()){
                find = true;
                auto xOp        = iter->second.back().first;
                ops.emplace_back(xOp);
            }
        }
        if(find == false){
            MNN_PRINT("not find input \n ");
        };
        return ops[index];
    }

    void NPUBackend::setOutputOps(const Op *op, vector<shared_ptr<ge::Operator>>&& HIAI_op,
                                  const std::vector<Tensor *> &outputs){
        if(op->type() == OpType_Slice || op->type() == OpType_TopKV2){
            for (size_t i = 0; i < op->outputIndexes()->size(); i++){
                auto index = op->outputIndexes()->data()[i];
                mSclipMap[index] = i;
            }
        }
        for (size_t i = 0; i < op->outputIndexes()->size(); i++){
            auto index = op->outputIndexes()->data()[i];
            vector<pair<shared_ptr<ge::Operator>, string>> ops;
            for (size_t j = 0; j < HIAI_op.size(); j++){
                ops.emplace_back(make_pair(HIAI_op[j], ""));
            }
            mGrapMap.insert(make_pair(index, ops));
        }

        MNNTensorList tensors;
        for (auto out: outputs)
        {
            bool isOutput = (TensorUtils::getDescribe(out)->usage 
                            ==Tensor::InsideDescribe::Usage::OUTPUT);
            if(isOutput == true){
                tensors.push_back(out);
            }
        }
        if(!tensors.empty()) {
            mOutGEOpMap.insert(make_pair(HIAI_op[HIAI_op.size()-1], tensors));
        }
    }

    NPURuntime::NPURuntime(const Backend::Info& info) {
        mInfo = info;

        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    NPURuntime::~NPURuntime() {}

    Backend* NPURuntime::onCreate(const BackendConfig* config) const {
        return new NPUBackend(this);
    }

    void NPURuntime::onGabageCollect(int level) {
        // nothing now
    }
    Runtime::CompilerType NPURuntime::onGetCompilerType() const {
        return Compiler_Origin;
    }

    struct NPUBackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            AUTOTIME;
            {
                shared_ptr<hiai::AiModelMngerClient> mgrClient = make_shared<hiai::AiModelMngerClient>();
                if(mgrClient.get() == nullptr){
                    MNN_ERROR("mgrClient.get() == NULL");
                    return nullptr;
                }
                
				auto ret = mgrClient->Init(nullptr);
                if (ret != hiai::AI_SUCCESS) {
                    MNN_ERROR("[NPU] AiModelMngerClient Init Failed!\n");
                    return nullptr;
                }
				
                const char* currentversion = mgrClient->GetVersion();
                if(currentversion != nullptr){
                    MNN_PRINT("[NPU] ddk currentversion : %s \n", currentversion);
                }else{
                    MNN_ERROR("[NPU] current version don't support, return nullptr\n");
                    return nullptr;
                }

                if(string(currentversion).compare("100.330.000.000") <= 0){
                    MNN_PRINT("[NPU] current version don't support,version=%s \n",currentversion);
                    return nullptr;
                }
            }

            return new NPURuntime(info);
        }

        virtual bool onValid(Backend::Info& info) const {
            return true;
        }
    };

    static const auto __npu_global_initializer = []() {
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_0, new NPUBackendCreator, true);
        return true;
    }();
}
