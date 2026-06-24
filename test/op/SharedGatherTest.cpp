//
//  SharedGatherTest.cpp
//  MNNTests
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

// SharedGather requires the int8/low-memory weight quant path. The
// corresponding executor (DenseConvInt8TiledExecutor::onClone -> SharedGather)
// is only selected by ConvolutionFloatFactory when MNN_LOW_MEMORY is enabled.
// Without it, the conv base falls back to DenseConvolutionTiledExecutor, which
// cannot serve SharedGather, so this test cannot validate the feature and must
// be skipped to avoid spurious failures in non-low-memory CI builds.
#ifdef MNN_LOW_MEMORY

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/IDSTEncoder.hpp"

using namespace MNN;
using namespace MNN::Express;

static std::shared_ptr<Executor::RuntimeManager> makeSharedGatherRuntime() {
    auto status = MNNTestSuite::get()->pStaus;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(status.precision);
    backendConfig.memory = BackendConfig::Memory_Low;
    ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(status.forwardType);
    config.backendConfig = &backendConfig;
    config.numThread = status.thread > 0 ? status.thread : 1;
    return std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(config),
                                                     Executor::RuntimeManager::destroy);
}

static VARP makeSharedConv(VARP input, const std::vector<float>& weight, int ic, int oc) {
    std::unique_ptr<OpT> conv(new OpT);
    conv->type = OpType_Convolution;
    conv->name = "shared_weight";
    conv->main.type = OpParameter_Convolution2D;
    conv->main.value = new Convolution2DT;
    auto conv2D = conv->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->kernelX = 1;
    conv2D->common->kernelY = 1;
    conv2D->common->strideX = 1;
    conv2D->common->strideY = 1;
    conv2D->common->dilateX = 1;
    conv2D->common->dilateY = 1;
    conv2D->common->group = 1;
    conv2D->common->inputCount = ic;
    conv2D->common->outputCount = oc;
    conv2D->bias.resize(oc, 0.0f);

    std::vector<float> scale;
    scale.reserve(oc * 2);
    for (int o = 0; o < oc; ++o) {
        scale.emplace_back(-1.0f + 0.05f * o);
        scale.emplace_back(0.125f + 0.01f * o);
    }
    IDSTEncoder::EncodeOptions options;
    options.bits = 4;
    conv2D->quanParameter = IDSTEncoder::encode(weight.data(), scale, ic, oc, true, nullptr, -8, options);

    auto expr = Expr::create(std::move(conv), {input});
    expr->setName("shared_weight");
    auto output = Variable::create(expr);
    output->setName("shared_weight");
    return output;
}

static VARP makeSharedGather(VARP indices, int ic, int oc) {
    std::unique_ptr<OpT> gather(new OpT);
    gather->type = OpType_GatherV2;
    gather->name = "shared_weight";
    gather->main.type = OpParameter_Input;
    gather->main.value = new InputT;
    gather->main.AsInput()->dims = {oc, ic};
    gather->main.AsInput()->dtype = DataType_DT_FLOAT;
    gather->main.AsInput()->dformat = MNN_DATA_FORMAT_NCHW;
    auto expr = Expr::create(std::move(gather), {indices});
    expr->setName("shared_weight");
    auto output = Variable::create(expr);
    output->setName("shared_weight");
    return output;
}

class SharedGatherTest : public MNNTestCase {
public:
    virtual ~SharedGatherTest() = default;
    virtual bool run(int precision) {
        const int ic = 64;
        const int oc = 8;
        std::vector<float> weight(oc * ic);
        for (int o = 0; o < oc; ++o) {
            float minValue = -1.0f + 0.05f * o;
            float step = 0.125f + 0.01f * o;
            for (int c = 0; c < ic; ++c) {
                weight[o * ic + c] = minValue + step * ((c + o) % 16);
            }
        }

        auto baseInput = _Input({1, ic, 1, 1}, NCHW);
        ::memset(baseInput->writeMap<float>(), 0, ic * sizeof(float));
        baseInput->unMap();
        auto convOutput = makeSharedConv(baseInput, weight, ic, oc);
        auto baseBuffer = Variable::save({convOutput});

        int indicesData[] = {3, 0, 7, 2};
        auto indicesInput = _Input({4}, NCHW, halide_type_of<int>());
        indicesInput->setName("x");
        auto gatherOutput = makeSharedGather(indicesInput, ic, oc);
        auto gatherBuffer = Variable::save({gatherOutput});

        auto runtime = makeSharedGatherRuntime();
        Module::Config baseConfig;
        baseConfig.rearrange = true;
        std::shared_ptr<Module> base(Module::load({}, {}, (const uint8_t*)baseBuffer.data(), baseBuffer.size(), runtime,
                                                  &baseConfig));
        if (!base) {
            MNN_ERROR("SharedGatherTest load base module failed!\n");
            return false;
        }
        Module::Config gatherConfig;
        gatherConfig.rearrange = true;
        gatherConfig.base = base.get();
        std::shared_ptr<Module> gather(Module::load({}, {}, (const uint8_t*)gatherBuffer.data(), gatherBuffer.size(),
                                                    runtime, &gatherConfig));
        if (!gather) {
            MNN_ERROR("SharedGatherTest load gather module failed!\n");
            return false;
        }
        auto runtimeIndices = _Input({4}, NCHW, halide_type_of<int>());
        ::memcpy(runtimeIndices->writeMap<int>(), indicesData, sizeof(indicesData));
        runtimeIndices->unMap();
        auto outputs = gather->onForward({runtimeIndices});
        if (outputs.empty() || outputs[0] == nullptr) {
            MNN_ERROR("SharedGatherTest forward failed!\n");
            return false;
        }
        auto output = outputs[0];
        std::vector<float> expected(4 * ic);
        for (int i = 0; i < 4; ++i) {
            ::memcpy(expected.data() + i * ic, weight.data() + indicesData[i] * ic, ic * sizeof(float));
        }
        if (!checkVector<float>(output->readMap<float>(), expected.data(), expected.size(), 0.02f)) {
            MNN_ERROR("SharedGatherTest failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(SharedGatherTest, "op/shared_gather");

#endif // MNN_LOW_MEMORY
