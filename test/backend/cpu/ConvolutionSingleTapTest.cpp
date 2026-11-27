// Tests for packed-weight ownership and dynamic dilated convolution shapes.
#include <cmath>
#include <memory>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/ConvolutionFloatFactory.h"

using namespace MNN;
using namespace MNN::Express;

#ifdef MNN_TEST_CPU_INTERNAL
namespace {
class AllocationBackend : public CPUBackend {
public:
    AllocationBackend(const CPURuntime* runtime, int failAt)
        : CPUBackend(runtime, BackendConfig::Precision_High, BackendConfig::Memory_Normal, MNN_FORWARD_CPU, 0),
          mFailAt(failAt) {
    }
    MemObj* onAcquire(const Tensor* tensor, StorageType storage) override {
        if (storage == STATIC && ++allocations == mFailAt) {
            return nullptr;
        }
        return CPUBackend::onAcquire(tensor, storage);
    }
    int allocations = 0;
private:
    int mFailAt;
};

static std::unique_ptr<flatbuffers::FlatBufferBuilder> convolutionOp() {
    OpT op;
    op.type = OpType_Convolution;
    op.main.type = OpParameter_Convolution2D;
    op.main.value = new Convolution2DT;
    auto conv = op.main.AsConvolution2D();
    conv->common.reset(new Convolution2DCommonT);
    conv->common->kernelX = conv->common->kernelY = 3;
    conv->common->dilateX = conv->common->dilateY = 36;
    conv->common->inputCount = 5;
    conv->common->outputCount = 7;
    conv->common->padMode = PadMode_SAME;
    conv->weight.resize(5 * 7 * 9, 0.125f);
    conv->bias.resize(7, 0.25f);
    std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder);
    builder->Finish(Op::Pack(*builder, &op));
    return builder;
}
}

class ConvolutionSingleTapAllocationTest : public MNNTestCase {
public:
    bool run(int precision) override {
        Backend::Info info;
        info.numThread = 1;
        CPURuntime runtime(info);
        auto model = convolutionOp();
        auto op = flatbuffers::GetRoot<Op>(model->GetBufferPointer());
        std::unique_ptr<Tensor> input(Tensor::createDevice<float>({1, 5, 33, 33}, Tensor::CAFFE_C4));
        std::unique_ptr<Tensor> output(Tensor::createDevice<float>({1, 7, 33, 33}, Tensor::CAFFE_C4));
        int count = 0;
        {
            AllocationBackend backend(&runtime, 0);
            std::unique_ptr<Execution> exe(ConvolutionFloatFactory::create({input.get()}, {output.get()}, op, &backend));
            if (!exe || !exe->valid()) {
                return false;
            }
            count = backend.allocations;
        }
        if (count == 0) {
            return false;
        }
        // Fail each real allocation, covering both full and reduced packed weights and biases.
        for (int failAt = 1; failAt <= count; ++failAt) {
            AllocationBackend backend(&runtime, failAt);
            std::unique_ptr<Execution> exe(ConvolutionFloatFactory::create({input.get()}, {output.get()}, op, &backend));
            if (exe && exe->valid()) {
                MNN_ERROR("Dilated convolution accepted failed allocation %d/%d\n", failAt, count);
                return false;
            }
            if (exe && exe->onClone(nullptr, op, nullptr)) {
                return false;
            }
        }
        return true;
    }
};

class ConvolutionSingleTapCloneTest : public MNNTestCase {
public:
    bool run(int precision) override {
        Backend::Info info;
        info.numThread = 1;
        CPURuntime runtime(info);
        auto model = convolutionOp();
        auto op = flatbuffers::GetRoot<Op>(model->GetBufferPointer());
        AllocationBackend backend(&runtime, 0);
        AllocationBackend cloneBackend(&runtime, 1);
        std::unique_ptr<Tensor> input(Tensor::createDevice<float>({1, 5, 33, 33}, Tensor::CAFFE_C4));
        std::unique_ptr<Tensor> output(Tensor::createDevice<float>({1, 7, 33, 33}, Tensor::CAFFE_C4));
        std::unique_ptr<Execution> exe(ConvolutionFloatFactory::create({input.get()}, {output.get()}, op, &backend));
        if (!exe || !exe->valid() || !exe->onClone(nullptr, op, nullptr)) {
            return false;
        }
        Execution* cloned = nullptr;
        const bool success = exe->onClone(&cloneBackend, op, &cloned);
        std::unique_ptr<Execution> copy(cloned);
        if (!success || !copy || !copy->valid() || cloneBackend.allocations != 0) {
            MNN_ERROR("Dilated convolution clone failed or repacked weights\n");
            return false;
        }
        exe.reset();
        if (!copy->onClone(nullptr, op, nullptr)) {
            return false;
        }
        return true;
    }
};

#endif

class ConvolutionSingleTapResizeTest : public MNNTestCase {
public:
    bool run(int precision) override {
        auto input = _Input({1, 5, 33, 33}, NCHW, halide_type_of<float>());
        input->setName("input");
        auto conv = _Conv(std::vector<float>(5 * 7 * 9, 0.125f), std::vector<float>(7, 0.25f), input,
                          {5, 7}, {3, 3}, SAME, {1, 1}, {36, 36}, 1, {0, 0}, false, false);
        auto output = _Convert(conv, NCHW);
        output->setName("output");
        auto model = Variable::save({output});
        for (bool rearrange : {false, true}) {
            Module::Config config;
            config.shapeMutable = true;
            config.rearrange = rearrange;
            std::unique_ptr<Module> original(Module::load({"input"}, {"output"},
                reinterpret_cast<const uint8_t*>(model.data()), model.size(), &config));
            if (!original || !check(original.get(), 33, 33)) {
                return false;
            }
            std::unique_ptr<Module> copy(Module::clone(original.get(), true));
            if (!copy) {
                return false;
            }
            // Different shapes on the original and clone catch shared mutable execution state.
            for (int size : {37, 33, 36, 73, 33}) {
                if (!check(original.get(), size, 33) || !check(copy.get(), 33, size)) {
                    return false;
                }
            }
            original.reset();
            if (!check(copy.get(), 37, 37) || !check(copy.get(), 33, 33)) {
                return false;
            }
        }
        return true;
    }
private:
    static bool check(Module* module, int height, int width) {
        auto input = _Input({2, 5, height, width}, NCHW, halide_type_of<float>());
        auto data = input->writeMap<float>();
        for (int i = 0; i < 2 * 5 * height * width; ++i) {
            data[i] = 1.0f;
        }
        auto result = module->onForward({input});
        if (result.size() != 1) {
            return false;
        }
        auto info = result[0]->getInfo();
        auto actual = result[0]->readMap<float>();
        if (!info || info->dim != std::vector<int>({2, 7, height, width}) || !actual) {
            return false;
        }
        for (int n = 0; n < 2 * 7; ++n) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Each in-bounds tap sums five inputs; off-centre weights are nonzero.
                    int taps = 0;
                    for (int dy : {-36, 0, 36}) {
                        for (int dx : {-36, 0, 36}) {
                            taps += y + dy >= 0 && y + dy < height && x + dx >= 0 && x + dx < width;
                        }
                    }
                    const float expected = 0.25f + taps * 5 * 0.125f;
                    if (!std::isfinite(actual[(n * height + y) * width + x]) ||
                        std::abs(actual[(n * height + y) * width + x] - expected) > 1e-4f) {
                        MNN_ERROR("Dilated convolution mismatch: shape=%dx%d y=%d x=%d\n", height, width, y, x);
                        return false;
                    }
                }
            }
        }
        return true;
    }
};

#ifdef MNN_TEST_CPU_INTERNAL
MNNTestSuiteRegister(ConvolutionSingleTapAllocationTest, "op/ConvolutionSingleTap/allocation");
MNNTestSuiteRegister(ConvolutionSingleTapCloneTest, "op/ConvolutionSingleTap/clone");
#endif
MNNTestSuiteRegister(ConvolutionSingleTapResizeTest, "op/ConvolutionSingleTap/resize");
