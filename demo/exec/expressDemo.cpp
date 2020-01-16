#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
#define UP_DIV(x) (((x)+3)/4)

static std::pair<VARP, VARP> _makeConvolution(int k, int ic, int oc, int size) {
    auto input = _Input({1, ic, size, size}, NC4HW4);
    return std::make_pair(input, _Conv(0.0f, 0.0f, input, {ic, oc}, {k, k}, SAME));
}
static std::pair<VARP, VARP> _makeGEMMByMatMul(int e, int l, int h) {
    auto a = _Input({e, l});
    std::vector<float> weight(l*h);
    auto b = _Const(weight.data(), {l, h});
    auto c = _MatMul(a, b);
    return std::make_pair(a, c);
}

static std::pair<VARP, VARP> _makeGEMMByConvolution(int e, int l, int h) {
    auto icC4 = UP_DIV(l);
    auto ocC4 = UP_DIV(h);

    auto input = _Input({1, icC4*4, 1, e});
    return std::make_pair(input, _Conv(0.0f, 0.0f, input, {icC4*4, ocC4*4}, {1, 1}));
}
static void _testConvolution() {
    std::vector<std::vector<int>> size = {
        {2, 3, 16, 224},
        {3, 3, 16, 224},
        {5, 3, 16, 224},
        {2, 16, 4, 224},
        {3, 16, 4, 224},
        {5, 16, 4, 224},
        {2, 16, 16, 224},
        {3, 16, 16, 224},
        {5, 16, 16, 224},
        {2, 64, 64, 112},
        {3, 64, 64, 112},
        {5, 64, 64, 112},
        {2, 512, 512, 4},
        {3, 512, 512, 4},
        {5, 512, 512, 4},
        {2, 512, 512, 16},
        {3, 512, 512, 16},
        {5, 512, 512, 16},
        {2, 512, 512, 32},
        {3, 512, 512, 32},
        {5, 512, 512, 32},
    };

    auto conv = _makeGEMMByConvolution(1024, 1024, 1024);
    for (int v=0; v<10; ++v) {
        conv.first->writeMap<float>();
        conv.first->unMap();
        conv.second->readMap<float>();
        conv.second->unMap();
    }
    for (int i=0; i<size.size(); ++i) {
        conv = _makeConvolution(size[i][0], size[i][1], size[i][2], size[i][3]);
        MNN_PRINT("%d, %d, %d, %d: ", size[i][0], size[i][1], size[i][2], size[i][3]);
        AUTOTIME;
        for (int v=0; v<10; ++v) {
            conv.first->writeMap<float>();
            conv.first->unMap();
            conv.second->readMap<float>();
            conv.second->unMap();
        }
    }
}

static void _testGEMM() {
    std::vector<std::vector<int>> size = {
        {64, 64, 64},
        {64, 64, 128},
        {128, 128, 128},
        {128, 128, 256},
        {256, 256, 256},
        {256, 256, 512},
        {512, 512, 512},
        {512, 512, 1024},
        {1024, 1024, 1024},
    };
    for (int i=0; i<size.size(); ++i) {
        auto x = size[i][0];
        auto y = size[i][1];
        auto z = size[i][2];
        auto flops = (float)x * (float)y * (float)z / 1024.0f / 1024.0f;
        FUNC_PRINT_ALL(flops, f);
    }

    auto conv = _makeGEMMByConvolution(1024, 1024, 1024);
    for (int v=0; v<10; ++v) {
        conv.first->writeMap<float>();
        conv.first->unMap();
        conv.second->readMap<float>();
        conv.second->unMap();
    }
    for (int i=0; i<size.size(); ++i) {
        conv = _makeGEMMByConvolution(size[i][0], size[i][1], size[i][2]);
        AUTOTIME;
        for (int v=0; v<10; ++v) {
            conv.first->writeMap<float>();
            conv.first->unMap();
            conv.second->readMap<float>();
            conv.second->unMap();
        }
    }
    for (int i=0; i<size.size(); ++i) {
        conv = _makeGEMMByMatMul(size[i][0], size[i][1], size[i][2]);
        AUTOTIME;
        for (int v=0; v<10; ++v) {
            conv.first->writeMap<float>();
            conv.first->unMap();
            conv.second->readMap<float>();
            conv.second->unMap();
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_ERROR("./expressDemo.out model_path type testTime\n");
        return 0;
    }
    auto modelFileName = argv[1];
    FUNC_PRINT_ALL(modelFileName, s);
    auto exe = Executor::getGlobalExecutor();
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_Low;
    MNNForwardType forwardType = MNN_FORWARD_CPU;
    if (argc >= 3) {
        forwardType = (MNNForwardType)atoi(argv[2]);
    }
    exe->setGlobalExecutorConfig(forwardType, config, 4);
    auto model = Variable::loadMap(modelFileName);
    auto inputOutput = Variable::getInputAndOutput(model);
    auto inputs = inputOutput.first;
    auto outputs = inputOutput.second;
    int testTime = 10;
    if (argc >= 4) {
        testTime = atoi(argv[3]);
    }
    Variable::save(Variable::mapToSequence(outputs), "temp.mnn");
    auto input = inputs.begin()->second;
    auto output = outputs.begin()->second;
    //input->resize({1, 224, 224, 3});
    auto inputInfo = input->getInfo();
    if (nullptr == inputInfo) {
        return 0;
    }
    {
        AUTOTIME;
        input = _ChangeInputFormat(input, NCHW);
        inputInfo = input->getInfo();
        if (output->getInfo()->order == NC4HW4) {
            output = _Convert(output, NCHW);
        }
    }
    auto outputInfo = output->getInfo();
    if (nullptr == outputInfo) {
        MNN_ERROR("Output Not valid\n");
        return 0;
    }
    auto size = outputInfo->size;
    exe->gc(Executor::FULL);
    //Test Speed
    if (testTime > 0){
        //Let the frequence up
        for (int i=0; i<3; ++i) {
            input->writeMap<float>();
            input->unMap();
            output->readMap<float>();
            output->unMap();
        }
        AUTOTIME;
        for (int i=0; i<testTime; ++i) {
            input->writeMap<float>();
            input->unMap();
            output->readMap<float>();
            output->unMap();
        }
    }
    {
        auto size = inputInfo->size;
        auto inputPtr = input->writeMap<float>();
        std::ifstream inputOs("input_0.txt");
        for (int i=0; i<size; ++i) {
            inputOs >> inputPtr[i];
        }
        input->unMap();
    }

    {
        auto outputPtr = output->readMap<float>();
        if (nullptr == outputPtr) {
            MNN_ERROR("Output Not valid read error\n");
            return 0;
        }
        std::ofstream outputOs("output.txt");
        for (int i=0; i<size; ++i) {
            outputOs << outputPtr[i] << "\n";
        }
        output->unMap();
    }

    return 0;
}
