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

#include <iostream>
#include "core/MNNFileUtils.h"

using namespace MNN;
using namespace MNN::Transformer;


class TensorRange {
public:
    TensorRange(int featureMapBit, int tensorIndex, std::string tmpDir) : mFeatureMapBit(featureMapBit), mTensorIndex(tensorIndex){
        mFeatureClampValue = (1 << mFeatureMapBit) - 1;
        mRange.first = -std::numeric_limits<float>().lowest();
        mRange.second = -mRange.first;

        if (!tmpDir.empty() && tmpDir.back() != '/' && tmpDir.back() != '\\') {
            tmpDir += '/';
        }
        mTmpPath = tmpDir + std::to_string(tensorIndex);
        mVisited = false;
    }
    ~TensorRange() {
        // Do nothing
    }

    void updateRange(Tensor* t){
        mVisited = true;
        auto mOriginTensor = t;
        auto tmpTensor = t;
        std::shared_ptr<Tensor> mHostTensor(new MNN::Tensor(t, MNN::Tensor::CAFFE));
        bool res = t->copyToHostTensor(mHostTensor.get());
        if (res) {
            tmpTensor = mHostTensor.get();
        }
        int size = tmpTensor->elementSize();
        float* dataPtr = tmpTensor->host<float>();
        auto minValue = mRange.first;
        auto maxValue = mRange.second;

        std::string indexStr = std::to_string(TensorUtils::getDescribe(t)->index);
        std::ofstream outputOs(mTmpPath.c_str(), std::ios::app); // append data

        for (int i = 0; i < size; ++i) {
            minValue = std::min(minValue, dataPtr[i]);
            maxValue = std::max(maxValue, dataPtr[i]);
            outputOs << dataPtr[i] << "\n";
        }
    }

    std::pair<float, int32_t> finishAndCompute(int quantizedToUint, int index){
        std::ifstream file(mTmpPath);
        std::vector<float> tempBuffer;
        float d_;
        int size = 0;
        while (file >> d_) {
            tempBuffer.push_back(d_);
            size++;
        }

        size_t minRank = static_cast<size_t>(size * 0);

        size_t maxRank = static_cast<size_t>(size * 1);

        if (maxRank >= size) maxRank = size - 1;
        if (minRank >= size) minRank = size - 1;

        std::nth_element(tempBuffer.begin(),
                         tempBuffer.begin() + minRank,
                         tempBuffer.end());
        float clip_min = tempBuffer[minRank];

        std::nth_element(tempBuffer.begin(),
                         tempBuffer.begin() + maxRank,
                         tempBuffer.end());
        float clip_max = tempBuffer[maxRank];

        mRange.first = ALIMIN(clip_min, mRange.first);
        mRange.second = ALIMAX(clip_max, mRange.second);

        mScale = (mRange.second - mRange.first) / mFeatureClampValue;
        mBias = static_cast<int>(roundf(mRange.first * mFeatureClampValue / (mRange.second - mRange.first)));
        if (quantizedToUint == 0) { // quantized to signed int
            float lowerThred = (float)(1 << (mFeatureMapBit - 1));
            mBias = static_cast<int>(roundf(-mRange.first * mFeatureClampValue / (mRange.second - mRange.first) - lowerThred));
        }
        return std::make_pair(mScale, mBias);
    }

    bool visited() {
        return mVisited;
    }


private:
    // <minVal, maxVal> for every channel for the Tensor
    std::pair<float, float> mRange;
    std::shared_ptr<MNN::Tensor> mHostTensor;

    float mScale;
    int32_t mBias = 0;
    float mFeatureClampValue = 127.0f;
    int32_t mFeatureMapBit = 8;
    int32_t mTensorIndex = -1;
    std::string mTmpPath = "";
    bool mVisited = false;
};
static void getFeature(std::map<int, std::shared_ptr<TensorRange>> &_featureInfo, Llm* llm, int bit, std::string tmpDir){
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (info->type() != "Convolution") {
            return true;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (TensorUtils::getDescribe(t)->index < 0) {
                continue;
            }
            if (_featureInfo.find(TensorUtils::getDescribe(t)->index) == _featureInfo.end() && t->getType().code == halide_type_float && TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::Usage::INPUT) {
                _featureInfo[TensorUtils::getDescribe(t)->index] = std::shared_ptr<TensorRange>(new TensorRange(bit, TensorUtils::getDescribe(t)->index, tmpDir));
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        if (info->type() != "Convolution") {
            return true;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (TensorUtils::getDescribe(t)->index < 0) {
                continue;
            }
            if (_featureInfo.find(TensorUtils::getDescribe(t)->index) == _featureInfo.end() && t->getType().code == halide_type_float && TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::Usage::OUTPUT) {
                _featureInfo[TensorUtils::getDescribe(t)->index] = std::shared_ptr<TensorRange>(new TensorRange(bit, TensorUtils::getDescribe(t)->index, tmpDir));
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
            llm->response(prompt, &std::cout, nullptr, max_token_number);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(prompt);
        }
    }
}

static void computeFeatureScaleKL(std::map<int, std::pair<float, int32_t>> &_scales,
                                  std::map<int, std::shared_ptr<TensorRange>> &_featureInfo,
                                  Llm* llm, const std::vector<std::string>& prompts, int max_token_number, int quantizedToUint) {
    _computeFeatureMapsRange(_featureInfo, llm, prompts, max_token_number);
    _scales.clear();
    for (auto& iter : _featureInfo) {
        _scales[iter.first] = iter.second->finishAndCompute(quantizedToUint, iter.first);
    }
}

static void _insertScale(MNN::NetT* _originalModel, std::map<int, std::pair<float, int32_t>> &_scales,
                         std::map<int, std::unique_ptr<MNN::TensorDescribeT>> &_tensorDescribes,
                         std::map<int, std::pair<float, int32_t>> tensorDescribesHasScaleIndex,
                          int featureBit, int weightBit, int blockSize) {
    float _featureClampValue = (float)((1 << (featureBit - 1)));
    auto type = MNN::DataType_DT_INT8;
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
        describe->quantInfo->max = _featureClampValue - 1;
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
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << " featureBit" << " dstFile " << "unsigned input" << "maxTokenForRange" << "tmpDirPath(deleted when finished)" << std::endl;
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

    int quantizedToUint = std::atoi(argv[5]);
    std::map<int, std::shared_ptr<TensorRange>> _featureInfo;
    std::map<int, std::pair<float, int32_t>> _scales;
    std::map<int, std::unique_ptr<MNN::TensorDescribeT>> _tensorDescribes;
    std::map<int, std::pair<float, int32_t>> tensorDescribesHasScaleIndex;

    int maxNewTokensToComputeRange = std::atoi(argv[6]);
    std::string tmpDir = argv[7];

    std::remove(tmpDir.c_str());
    MNNCreateDir(tmpDir.c_str());

    getFeature(_featureInfo, llm.get(), featureBit, tmpDir);
    computeFeatureScaleKL(_scales, _featureInfo, llm.get(), prompts, maxNewTokensToComputeRange, quantizedToUint);

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
