#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "llm/llm.hpp"
#include "llmconfig.hpp"
using namespace MNN::Express;
static MNN::Express::VARP _CrossEntropy(std::vector<MNN::Express::VARP> inputs, int ignore_index) {
    auto shape = _Shape(inputs[0], true), oneV = _Unsqueeze(_Scalar<int>(1), {0}), classes = _Slice(shape, oneV, oneV);
    auto mask = _OneHot(inputs[1], classes, _Scalar<float>(1), _Scalar<float>(0), 1);
    mask = mask * _Cast<float>(_Unsqueeze(_NotEqual(inputs[1], _Scalar<int>(ignore_index)), {1}));
    
    auto log_prob = inputs[0];
    log_prob = _Log(_Softmax(inputs[0], 1));
    auto temp = log_prob;
    auto output = _ReduceSum(mask * _Negative(temp), {1}, false);
    output = _ReduceMean(output);
    return output;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./ppl_eval model/config.json wiki_output\n");
        return 0;
    }
    auto llmPath = argv[1];
    auto textPath = argv[2];
    FUNC_PRINT_ALL(llmPath, s);
    FUNC_PRINT_ALL(textPath, s);
    std::shared_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(llmPath));
    {
        AUTOTIME;
        llm->set_config("{\"all_logits\":true, \"use_template\":false}");
        auto res = llm->load();
        if (!res) {
            MNN_ERROR("Load LLM error\n");
            return 0;
        }
    }
    std::string promptPath = std::string(textPath) + "/prompt.txt";
    std::vector<int> inputIds;
    {
        AUTOTIME;
        std::ifstream is(promptPath.c_str());
        if (is.fail()) {
            MNN_ERROR("Load prompt error\n");
            return 0;
        }
        std::ostringstream os;
        os << is.rdbuf();
        inputIds = llm->tokenizer_encode(os.str());
    }
    int ignore_index = -100;
    std::shared_ptr<MNN::Express::Module> cross;
    {
        auto x = _Input({}, NCHW);
        auto y = _Input({}, NCHW, halide_type_of<int>());
        x->setName("x");
        y->setName("y");
        auto z = _CrossEntropy({x, y}, ignore_index);
        z->setName("z");
        auto buffer = Variable::save({z});
        cross.reset(Module::load({"x", "y"}, {"z"}, (uint8_t*)buffer.data(), buffer.size()));
    }
    size_t stride = 512;
    size_t contextLength = stride + stride / 2;
    std::shared_ptr<MNN::Transformer::LlmConfig> lmConfig(new MNN::Transformer::LlmConfig(llmPath));
    if (lmConfig->config_.document.HasMember("chunk_limits")) {
        contextLength = lmConfig->config_.document["chunk_limits"].GetArray().begin()->GetInt();
        stride = (contextLength / 3) * 2;
    } else if (lmConfig->config_.document.HasMember("chunk")) {
        contextLength = lmConfig->config_.document["chunk"].GetInt();
        stride = (contextLength / 3) * 2;
    }
    FUNC_PRINT(contextLength);
    FUNC_PRINT(stride);
    auto seqLen = inputIds.size();
    size_t prevEnd = 0;
    
    float lossSum = 0.0f;
    int lossNumber = 0;
    for (size_t begin = 0; begin < seqLen; begin += stride) {
        auto end = std::min(begin + contextLength, seqLen);
        std::vector<int> chunkIds(end-begin);
        ::memcpy(chunkIds.data(), inputIds.data() + begin, chunkIds.size() * sizeof(int));
        llm->reset();
        auto logits = llm->forward(chunkIds);
        logits = MNN::Express::_Squeeze(logits, {0});
        auto trgLen = end - prevEnd;
        if (prevEnd != 0) {
            trgLen += 1;
        }
        std::vector<int> starts = {(int)(chunkIds.size() - trgLen), 0};
        std::vector<int> size = {(int)trgLen-1, -1};
        auto startVar = MNN::Express::_Const(starts.data(), {2}, MNN::Express::NCHW, halide_type_of<int>());
        auto sizeVar = MNN::Express::_Const(size.data(), {2}, MNN::Express::NCHW, halide_type_of<int>());
        logits = MNN::Express::_Slice(logits, startVar, sizeVar);
        auto target = _Const(chunkIds.data() + starts[0] + 1, {(int)trgLen - 1}, NCHW, halide_type_of<int>());
        auto loss = cross->onForward({logits, target})[0]->readMap<float>()[0];
        lossSum+=loss;
        lossNumber++;
        prevEnd = end;
        MNN_PRINT("Compute: %d/%d, loss=%f\n", begin, seqLen, loss);
        if (end == seqLen) {
            break;
        }
    }
    MNN_PRINT("Perplexity: %f\n", expf(lossSum / (float)lossNumber));

    return 0;
}
