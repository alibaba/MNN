//
//  NPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBackend.hpp"
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
#include <android/log.h>

//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

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

        printf("batch:%d,channel:%d,area:%d \n",batch,channel,area);
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

    bool WriteToBufferFile(ge::Buffer& buffer,std::string om_file_path)
    {
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!",om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(buffer.data(), 1, buffer.size(), fp);
        if (write_size != buffer.size()) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
        return true;
    }
    
    bool WriteToOMFile(domi::ModelBufferData om_model_buff,std::string om_file_path)
    {
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!",om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
        if (write_size != om_model_buff.length) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
        return true;
    }

    shared_ptr<hiai::AiModelMngerClient> LoadModelSync(domi::ModelBufferData modelBufferData, string model_name)
    {
        shared_ptr<hiai::AiModelMngerClient> mngerClient = make_shared<hiai::AiModelMngerClient>();
        if (mngerClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client make_shared error.");
            return nullptr;
        }

        int ret = mngerClient->Init(nullptr);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Manager Init Failed.");
            return nullptr;
        }

        shared_ptr<hiai::AiModelBuilder> mcbuilder = make_shared<hiai::AiModelBuilder>(mngerClient);
        hiai::MemBuffer* buffer = mcbuilder->InputMemBufferCreate(modelBufferData.data, modelBufferData.length);
        if (buffer == nullptr) {
            MNN_ERROR("[NPU] create MemBuffer failed");
            return nullptr;
        }

        shared_ptr<hiai::AiModelDescription> desc = make_shared<hiai::AiModelDescription>(model_name, 3, 0, 0, 0);
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        MNN_PRINT("[NPU] loadModel %s \n", desc->GetName().c_str());
        vector<shared_ptr<hiai::AiModelDescription>> model_desc;
        model_desc.push_back(desc);


        ret = mngerClient->Load(model_desc);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Load Failed.");
            mngerClient = nullptr;
        }

        mcbuilder->MemBufferDestroy(buffer);
        return mngerClient;
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
    }
    NPUBackend::~NPUBackend() {

    }

    void NPUBackend::setNetworkInput(const std::vector<Tensor *> &inputs, const Op* op){
        Tensor *inputTensor = inputs[0];

        MNN_PRINT("op name : %s \n op type : %s \n", op->name()->c_str(), EnumNameOpType(op->type()));

        for (size_t i = 0; i < inputs.size(); ++i){
            auto input = inputs[i];
            MNN_PRINT("\n");
            MNN_PRINT("in nchw : %d, %d, %d, %d \n", input->batch(), input->channel(), input->height(), input->width());
            for (size_t i = 0; i < input->buffer().dimensions; i++){
                MNN_PRINT("%d , ", input->buffer().dim[i].extent);
            }
            MNN_PRINT("\n");
        }

        // for (size_t i = 0; i < outputs.size(); i++){
        //     auto output = outputs[i];
        //     MNN_PRINT("\n");
        //     MNN_PRINT("out nchw : %d, %d, %d, %d \n", output->batch(), output->channel(), output->height(), output->width());
        //     for (size_t i = 0; i < output->buffer().dimensions; i++){
        //         MNN_PRINT("%d , ", output->buffer().dim[i].extent);
        //     }
        //     MNN_PRINT("\n");
        // }

        auto inputIndex = op->inputIndexes()->data()[0];
        auto outputIndex = op->outputIndexes()->data()[0];
        bool isInput = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        if (isInput && mGrapMap.find(inputIndex) == mGrapMap.end()) {
            auto opName = string("input") + to_string(inputIndex);
            shared_ptr<ge::op::Data> data(new ge::op::Data(opName));    
            MNN_PRINT("input format : %d \n", TensorUtils::getDescribe(inputTensor)->dimensionFormat);
            MNN_PRINT("shape : [%d, %d, %d, %d] \n", inputTensor->buffer().dim[0].extent, inputTensor->buffer().dim[1].extent, inputTensor->buffer().dim[2].extent, inputTensor->buffer().dim[3].extent);
            auto shape = tensorShapeFormat(inputTensor);
            ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT); 
            data->update_input_desc_x(desc);
            
            // map
            vector<pair<shared_ptr<ge::Operator>, string>> ops;
            ops.emplace_back(make_pair(data, ""));  
            mGrapMap.insert(make_pair(inputIndex, ops));
            std::pair<int, std::vector<ge::Operator>> item(outputIndex, {*data.get()});
            mInputOps.insert(item);
        }
    }

    Execution* NPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        
        if (iter == map->end()) {
            MNN_PRINT("[NPU] Don't support type %d, %s\n", op->type(), op->name()->c_str());
            return nullptr;
        }

        auto exe = iter->second->onCreate(inputs, outputs, op, this);

        if (nullptr == exe) {
            MNN_PRINT("[NPU] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
            return nullptr;
        }

        return exe;
    }

    void NPUBackend::NPUBackend::onExecuteBegin() const {
    }
    
    void NPUBackend::onExecuteEnd() const {
        process(0);
    }

    bool NPUBackend::onAcquireBuffer(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        if(isInputCopy){
            mInputMap.insert(make_pair((unsigned long)tensor, mInputMap.size()));
        }
        if(isOutputCopy){
            mOutputMap.insert(make_pair((unsigned long)tensor, mOutputMap.size()));
        }
        return true;
    }

    bool NPUBackend::onReleaseBuffer(const Tensor* tensor, StorageType storageType) {
        ((Tensor*)tensor)->buffer().device = uint64_t(0);
        return true;
    }

    bool NPUBackend::onClearBuffer() {
        return true;
    }

    void NPUBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {

        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

        if(isConst){ return; }
        
        if (isInputCopy) {
            shared_ptr<Tensor> tmpTensor(new Tensor(dstTensor, Tensor::DimensionType::CAFFE, true));

            tensorConvert(srcTensor, tmpTensor.get());
            auto index = mInputMap.find((unsigned long)(const_cast<Tensor*>(dstTensor)));
            MNN_ASSERT(index != mInputMap.end());
            shared_ptr<hiai::AiTensor> input = mInputTensors[index->second];
            memcpy(input->GetBuffer(), tmpTensor->host<float>(), (size_t)tmpTensor->size());

        } else if(isOutputCopy){
            auto index = mOutputMap.find((unsigned long)(const_cast<Tensor*>(srcTensor)));
            MNN_ASSERT(index != mOutputMap.end());
            shared_ptr<hiai::AiTensor> output = mOutputTensors[index->second];
            auto tmpShape = tensorShapeFormat(srcTensor);
            vector<int> srcShape = {(int)tmpShape[0],(int)tmpShape[1],(int)tmpShape[2],(int)tmpShape[3]};
            shared_ptr<Tensor> tmpTensor(Tensor::create(srcShape,halide_type_of<float>(),
                                                        (void*)(output->GetBuffer()), 
                                                        Tensor::DimensionType::CAFFE));// nchw
            auto shape = output->GetTensorDimension(); 
            tensorConvert(tmpTensor.get(), dstTensor);
        }

        // setTensorIndex();
    }

    void NPUBackend::onResizeBegin() {
        mGrapMap.clear();
        mGrapIOMap.clear();
        mInputOps.clear();
        mOutputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mSclipMap.clear();
    }

    void NPUBackend::onResizeEnd() {
        bulidIRModelAndLoad();
    }


    int NPUBackend::getInOutTensorInfo(string modelName) {
        if (mMgrClient == nullptr) {
            return -1;
        }

        int ret = mMgrClient->GetModelIOTensorDim(modelName, mInputDimension, mOutputDimension);
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("[NPU] Get model IO Tensor failed：%d \n", ret);
            return -1;
        }

        MNN_PRINT("mInputDimension : %lu , mOutputDimension : %lu \n", mInputDimension.size(), mOutputDimension.size());

        for (auto in_dim : mInputDimension)
        {
            shared_ptr<hiai::AiTensor> input = make_shared<hiai::AiTensor>();
            input->Init(&in_dim);
            mInputTensors.push_back(input);
        }
        for (auto out_dim : mOutputDimension)
        {
            shared_ptr<hiai::AiTensor> output = make_shared<hiai::AiTensor>();
            output->Init(&out_dim);
            mOutputTensors.push_back(output);
        }

        return 0;
    }

    int i = 0;

    void NPUBackend::bulidIRModelAndLoad() {
        MNN_PRINT("mInputOps : %lu \n", mInputOps.size());
        std::vector<ge::Operator> inputs;
        for (auto input : mInputOps){
            inputs.push_back(input.second[0]);
        }

        string graphName = string("Graph1");
        string version = string("model_v000011");
        string modelName = to_string(0);
        mModelName.push_back(modelName);
        ge::Graph graph(graphName);
        graph.SetInputs(inputs).SetOutputs(mOutputOps);

        ge::Model model(modelName, version);
        model.SetGraph(graph);

        
        domi::HiaiIrBuild ir_build;
        domi::ModelBufferData om_model_buff;

        ge::Buffer buffer;
        ge::GraphErrCodeStatus geret = model.Save(buffer);
        if(geret != 0) {
            MNN_ERROR("[NPU] Model save failed \n");
        }

        //WriteToBufferFile(buffer, "/data/local/tmp/test.irpb");

        bool createBufferSuc = ir_build.CreateModelBuff(model, om_model_buff);

        if (!createBufferSuc) {
            MNN_ERROR("[NPU] Create Model Buff failed \n");
        }
        bool buildIRSuc = ir_build.BuildIRModel(model, om_model_buff);
        if(!buildIRSuc){
            MNN_ERROR("[NPU] IR model build failed  \n");
        }

        //WriteToOMFile(om_model_buff, "/data/local/tmp/test.om");

        mMgrClient = LoadModelSync(om_model_buff, modelName);

        if (mMgrClient==nullptr) {
            MNN_ERROR("[NPU] Model Manager Client is null \n");
            ir_build.ReleaseModelBuff(om_model_buff);
        }

        ir_build.ReleaseModelBuff(om_model_buff);

        int result = getInOutTensorInfo(modelName);

        MNN_ASSERT(result == 0);
               
    }

    int NPUBackend::process(int modelIndex) const {

        hiai::AiContext context;
        string key = "model_name";
        string value = to_string(modelIndex);
        context.AddPara(key, value);

        int istamp;

        AUTOTIME;
        int ret = mMgrClient->Process(context,*(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mInputTensors)), *(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mOutputTensors)), 1000, istamp);

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

    void NPUBackend::setOutputIOOps(const Op *op, vector<shared_ptr<ge::OpIO>>&& HIAI_op){
        for (size_t i = 0; i < op->outputIndexes()->size(); i++){
            auto index = op->outputIndexes()->data()[i];
            vector<pair<shared_ptr<ge::OpIO>, string>> ops;
            for (size_t j = 0; j < HIAI_op.size(); j++){
                ops.emplace_back(make_pair(HIAI_op[j], ""));
            }
            mGrapIOMap.insert(make_pair(index, ops));
        }
    }

    void NPUBackend::setOutputOps(const Op *op, vector<shared_ptr<ge::Operator>>&& HIAI_op){
        if(op->type() == OpType_Slice){
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

    Backend* NPURuntime::onCreate() const {
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

                if(string(currentversion).compare("100.320.000.000") <= 0){
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
