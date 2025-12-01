#include "llm/llm.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>
#include "core/TensorUtils.hpp"
#include "flatbuffers/util.h"
#include "llmconfig.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/ConvolutionCommon.hpp"

using namespace MNN;
using namespace MNN::Transformer;


class TensorRange {
public:
    TensorRange(float featureClampValue) : mFeatureClampValue(featureClampValue){
    }
    ~TensorRange() {
        // Do nothing
    }

    void updateRange(Tensor* t){
        if (mUpdatedRangeFlags) {
            return;
        }
        auto mOriginTensor = t;
        mUpdatedRangeFlags = true;
        auto tmpTensor = mOriginTensor;
        std::shared_ptr<Tensor> mHostTensor(new MNN::Tensor(mOriginTensor, MNN::Tensor::CAFFE));
        bool res = mOriginTensor->copyToHostTensor(mHostTensor.get());
        if (res) {
            tmpTensor = mHostTensor.get();
        }
        int size = tmpTensor->elementSize();
        float* dataPtr = tmpTensor->host<float>();
        auto minValue = mRange.first;
        auto maxValue = mRange.second;
        for (int i = 0; i < size; ++i) {
            minValue = std::min(minValue, dataPtr[i]);
            maxValue = std::max(maxValue, dataPtr[i]);
        }
        mRange.first = minValue;
        mRange.second = maxValue;
        mVisited = true;
    }

    std::pair<float, int8_t> finishAndCompute(){
        auto maxValue         = std::max(fabsf(mRange.second), fabsf(mRange.first));
        mZeroPoint = 0;
        mScale = maxValue / mFeatureClampValue;
        
        return std::make_pair(mScale, mZeroPoint);
    }
    
    void resetUpdatedRangeFlags() {
        mUpdatedRangeFlags = false;
    }
    
    bool visited() {
        return mVisited;
    }

    void setVisited(bool visited) {
        mVisited = visited;
    }


private:
    // <minVal, maxVal> for every channel for the Tensor
    std::pair<float, float> mRange;
    std::shared_ptr<MNN::Tensor> mHostTensor;

    // the Tensor
    bool mUpdatedRangeFlags = false;
    bool mVisited = false;
    float mScale;
    int8_t mZeroPoint = 0;
    float mFeatureClampValue = 127.0f;
};
static void getFeature(std::map<int, std::shared_ptr<TensorRange>> &_featureInfo, Llm* llm, int bit){
    float _featureClampValue = (float)((1 << (bit - 1)) - 1);
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (TensorUtils::getDescribe(t)->index < 0) {
                continue;
            }
            if (_featureInfo.find(TensorUtils::getDescribe(t)->index) == _featureInfo.end() && t->getType().code == halide_type_float && TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::Usage::INPUT) {
                _featureInfo[TensorUtils::getDescribe(t)->index] = std::shared_ptr<TensorRange>(new TensorRange(_featureClampValue));
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (TensorUtils::getDescribe(t)->index < 0) {
                continue;
            }
            if (_featureInfo.find(TensorUtils::getDescribe(t)->index) == _featureInfo.end() && t->getType().code == halide_type_float && TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::Usage::OUTPUT) {
                _featureInfo[TensorUtils::getDescribe(t)->index] = std::shared_ptr<TensorRange>(new TensorRange(_featureClampValue));
            }
        }
        return true;
    };
    
    Express::ExecutorScope::Current()->setCallBack(std::move(before), std::move(after));
    llm->tuning(OP_ENCODER_NUMBER, {1});
}

static void _computeFeatureMapsRange(std::map<int, std::shared_ptr<TensorRange>> &_featureInfo,
                                                  Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    auto context = llm->getContext();
    for (int i = 0; i < prompts.size(); i++) {
        llm->reset();
        auto prompt = prompts[i];
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }
        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                             const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                auto weakPtr = std::weak_ptr<Tensor::InsideDescribe::NativeInsideDescribe>(TensorUtils::getDescribeOrigin(t)->mContent);
                if (_featureInfo.find(TensorUtils::getDescribe(t)->index) != _featureInfo.end()) {
                    if (_featureInfo[TensorUtils::getDescribe(t)->index]->visited() == false) {
                        _featureInfo[TensorUtils::getDescribe(t)->index]->updateRange(t);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (TensorUtils::getDescribe(t)->index < 0) {
                    continue;
                }
                if (_featureInfo.find(TensorUtils::getDescribe(t)->index) != _featureInfo.end()) {
                    if (_featureInfo[TensorUtils::getDescribe(t)->index]->visited() == false) {
                        _featureInfo[TensorUtils::getDescribe(t)->index]->updateRange(t);
                    }
                }
            }
            return true;
        };
        Express::ExecutorScope::Current()->setCallBack(std::move(before), std::move(after));
        if (max_token_number >= 0) {
            llm->response(prompt, &std::cout, nullptr, 0);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(prompt);
        }
    }
}

static void computeFeatureScaleKL(std::map<int, std::pair<float, int8_t>> &_scales,
                                  std::map<int, std::shared_ptr<TensorRange>> &_featureInfo,
                                  Llm* llm, const std::vector<std::string>& prompts, int max_token_number) {
    _computeFeatureMapsRange(_featureInfo, llm, prompts, max_token_number);
    _scales.clear();
    for (auto& iter : _featureInfo) {
        _scales[iter.first] = iter.second->finishAndCompute();
    }
}

static void _insertScale(MNN::NetT* _originalModel, std::map<int, std::pair<float, int8_t>> &_scales,
                         std::map<int, std::unique_ptr<MNN::TensorDescribeT>> &_tensorDescribes,
                         std::map<int, std::pair<float, int8_t>> tensorDescribesHasScaleIndex,
                          int featureBit, int weightBit, int blockSize) {
    float _featureClampValue = (float)((1 << (featureBit - 1)) - 1);
    auto type = MNN::DataType_DT_INT8;
    auto _weightQuantizeMethod = "MAX_ABS";
    if(featureBit == 16){
        type = MNN::DataType_DT_INT16;
    }
    std::set<OpType> propagateOpTypes = { OpType_Raster, OpType_ReLU, OpType_ReLU6, OpType_Pooling,
                                              OpType_Interp, OpType_CropAndResize, OpType_ROIPooling};
    for (auto& op : _originalModel->oplists) {
        const auto opType = op->type;
        
        if(propagateOpTypes.find(opType) != propagateOpTypes.end()){
            bool needErase = false;
            for(int id = 0; id < op->inputIndexes.size() && needErase == false; ++id){
                auto iter = tensorDescribesHasScaleIndex.find(op->inputIndexes[id]);
                if(iter != tensorDescribesHasScaleIndex.end()){
                    needErase = true;
                }
            }
            for(int id = 0; id < op->outputIndexes.size() && needErase == false; ++id){
                auto iter = tensorDescribesHasScaleIndex.find(op->outputIndexes[id]);
                if(iter != tensorDescribesHasScaleIndex.end()){
                    needErase = true;
                }
            }
            if(needErase){
                for(int id = 0; id < op->inputIndexes.size(); ++id){
                    auto iter = _scales.find(op->inputIndexes[id]);
                    if(iter != _scales.end()){
                        _scales.erase(iter);
                    }
                }
                for(int id = 0; id < op->outputIndexes.size(); ++id){
                    auto iter = _scales.find(op->outputIndexes[id]);
                    if(iter != _scales.end()){
                        _scales.erase(iter);
                    }
                }
            }
        }
    }
    for (const auto iter :  _scales) {
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        auto index = iter.first;
        
        describe->index = index;
        describe->quantInfo.reset(new MNN::TensorQuantInfoT);
        describe->quantInfo->scale = iter.second.first;
        describe->quantInfo->zero = iter.second.second;
        describe->quantInfo->type = type;
        describe->quantInfo->min = -1 * _featureClampValue;
        describe->quantInfo->max = 1 * _featureClampValue;
        auto dstiter = _tensorDescribes.find(index);
        if (dstiter == _tensorDescribes.end()) {
            _tensorDescribes.insert(std::make_pair(index, std::move(describe)));
        } else {
            dstiter->second->quantInfo = std::move(describe->quantInfo);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << " featureBit int" << " dstFile "<< std::endl;
        return 0;
    }
    std::string prompt_file = argv[2];
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config(R"({"tmp_path":"tmp"})");
    llm->set_config(R"({"enable_debug":true})");
    
    //load llm model
    llm->load();
    
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.empty()) {
            continue;
        }
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 0;
    }
    
    int featureBit = std::atoi(argv[3]);
    int weightBit = 8;
    int blockSize = 1;
    std::string _destModelFile = argv[4];
    std::map<int, std::shared_ptr<TensorRange>> _featureInfo;
    std::map<int, std::pair<float, int8_t>> _scales;
    std::map<int, std::unique_ptr<MNN::TensorDescribeT>> _tensorDescribes;
    std::map<int, std::pair<float, int8_t>> tensorDescribesHasScaleIndex;
    getFeature(_featureInfo, llm.get(), featureBit);
    computeFeatureScaleKL(_scales, _featureInfo, llm.get(), prompts, 32);
    
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    std::string llmModelPath = config->llm_model();
    std::unique_ptr<MNN::NetT> netT;
    std::shared_ptr<MNN::Interpreter> netC(MNN::Interpreter::createFromFile(llmModelPath.c_str()), MNN::Interpreter::destroy);
    if (nullptr == netC.get()) {
        return 0;
    }
    netT = MNN::UnPackNet(netC->getModelBuffer().first);
    for(auto &iter : netT.get()->extraTensorDescribe){
        tensorDescribesHasScaleIndex[iter->index] = {iter->quantInfo->scale, iter->quantInfo->zero};
    }
    _insertScale(netT.get(), _scales, _tensorDescribes, tensorDescribesHasScaleIndex, featureBit, weightBit, blockSize);
    
    for (auto& iter : _tensorDescribes) {
        // 保留原来的feature scale量化参数
        if(tensorDescribesHasScaleIndex.find(iter.second->index) != tensorDescribesHasScaleIndex.end()){
            continue;
        }
        netT.get()->extraTensorDescribe.emplace_back(std::move(iter.second));
    }
    _tensorDescribes.clear();
    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, netT.get());
        builderOutput.Finish(len);
        std::ofstream output(_destModelFile, std::ofstream::binary);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}
