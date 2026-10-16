// Focused ARM NX integration test; also runs against ARM's compute emulation layers.
#include "component/VulkanArmNx.hpp"
#include "component/VulkanBuffer.hpp"
#include "component/VulkanCommandPool.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "core/IDSTEncoder.hpp"

using namespace MNN;
using namespace MNN::Express;

static bool lowLevel(const VulkanDevice& device, uint32_t m, uint32_t k, uint32_t n, bool int8) {
    auto graph = VulkanArmNx::createMatMul(device, m, k, n, int8);
    if (!graph)
        return false;
    VulkanMemoryPool pool(device, false);
    const auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const size_t aBytes = (m * k * (int8 ? 1 : 4) + 3) & ~size_t(3);
    const size_t bBytes = (k * n * (int8 ? 1 : 4) + 3) & ~size_t(3);
    VulkanBuffer a(pool, true, aBytes, nullptr, usage, VK_SHARING_MODE_EXCLUSIVE, memory);
    VulkanBuffer b(pool, true, bBytes, nullptr, usage, VK_SHARING_MODE_EXCLUSIVE, memory);
    VulkanBuffer c(pool, true, m * n * 4, nullptr, usage, VK_SHARING_MODE_EXCLUSIVE, memory);
    VulkanCommandPool commandPool(device);
    std::unique_ptr<VulkanCommandPool::Buffer> cmd(commandPool.allocBuffer());
    // Replay the same recorded commands with changing inputs to exercise persistent session state.
    cmd->begin(0);
    graph->encode(cmd->get(), a.buffer(), 0, b.buffer(), 0, c.buffer(), 0);
    cmd->end();
    for (int iteration = 0; iteration < 5; ++iteration) {
        auto* aData = a.map();
        auto* bData = b.map();
        std::memset(aData, 0, aBytes);
        std::memset(bData, 0, bBytes);
        for (uint32_t i = 0; i < m * k; ++i) {
            const int value = (static_cast<int>(i) + iteration) % 9 - 4;
            if (int8)
                static_cast<int8_t*>(aData)[i] = value;
            else
                static_cast<float*>(aData)[i] = value * 0.25f;
        }
        for (uint32_t i = 0; i < k * n; ++i) {
            const int value = (static_cast<int>(i) * 3 + iteration) % 7 - 3;
            if (int8)
                static_cast<int8_t*>(bData)[i] = value;
            else
                static_cast<float*>(bData)[i] = value * 0.5f;
        }
        a.unmap();
        b.unmap();
        commandPool.submitAndWait(cmd->get());
        const auto* result = c.map();
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t column = 0; column < n; ++column) {
                float expected = 0;
                for (uint32_t inner = 0; inner < k; ++inner) {
                    const int av = (static_cast<int>(row * k + inner) + iteration) % 9 - 4;
                    const int bv = (static_cast<int>(inner * n + column) * 3 + iteration) % 7 - 3;
                    expected += av * bv * (int8 ? 1.0f : 0.125f);
                }
                const auto offset = row * n + column;
                float actual =
                    int8 ? static_cast<const int32_t*>(result)[offset] : static_cast<const float*>(result)[offset];
                if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-5f) {
                    MNN_ERROR("ARM NX graph mismatch: %ux%ux%u [%u,%u], expected=%f actual=%f\n", m, k, n, row, column,
                              expected, actual);
                    c.unmap();
                    return false;
                }
            }
        }
        c.unmap();
    }
    return true;
}

static bool expression(bool requireArmNx, bool batch, bool transpose, BackendConfig::PrecisionMode precision) {
    BackendConfig config;
    config.precision = precision;
    auto executor =
        Executor::newExecutor(MNN_FORWARD_VULKAN, config, MNN_GPU_TUNING_NONE | (batch ? MNN_GPU_RECORD_BATCH : 0));
    ExecutorScope scope(executor);
    auto a = _Input({3, 5}, NCHW);
    auto b = _Input(transpose ? std::vector<int>{7, 5} : std::vector<int>{5, 7}, NCHW);
    auto result = _MatMul(_Square(a), b, false, transpose) + _Scalar<float>(0.25f);
    a->setName("a");
    b->setName("b");
    result->setName("output");
    const auto model = Variable::save({result});
    Module::Config moduleConfig;
    moduleConfig.shapeMutable = true;
    std::shared_ptr<Module> module(Module::load({"a", "b"}, {"output"}, reinterpret_cast<const uint8_t*>(model.data()),
                                                model.size(), &moduleConfig),
                                   Module::destroy);
    if (!module)
        return false;
    const auto before = VulkanArmNx::recordedDispatchCount();
    // Reuse the expression across shape changes and repeated forwards.
    for (int iteration = 0; iteration < 4; ++iteration) {
        int m = iteration < 2 ? 3 : 8, k = iteration < 2 ? 5 : 4, n = iteration < 2 ? 7 : 8;
        a->resize({m, k});
        b->resize(transpose ? std::vector<int>{n, k} : std::vector<int>{k, n});
        auto* aData = a->writeMap<float>();
        auto* bData = b->writeMap<float>();
        for (int i = 0; i < m * k; ++i)
            aData[i] = ((i + iteration) % 9 - 4) * 0.25f;
        for (int i = 0; i < k * n; ++i)
            bData[i] = ((i * 3 + iteration) % 7 - 3) * 0.5f;
        auto outputs = module->onForward({a, b});
        if (outputs.size() != 1)
            return false;
        const auto* output = outputs[0]->readMap<float>();
        if (!output)
            return false;
        for (int row = 0; row < m; ++row)
            for (int column = 0; column < n; ++column) {
                float expected = 0.25f;
                for (int inner = 0; inner < k; ++inner) {
                    const float av = ((row * k + inner + iteration) % 9 - 4) * 0.25f;
                    const int bi = transpose ? column * k + inner : inner * n + column;
                    const float bv = ((bi * 3 + iteration) % 7 - 3) * 0.5f;
                    expected += av * av * bv;
                }
                if (!std::isfinite(output[row * n + column]) ||
                    std::fabs(output[row * n + column] - expected) > 1e-4f) {
                    MNN_ERROR(
                        "MNN mixed graph mismatch: batch=%d transpose=%d precision=%d iteration=%d [%d,%d] expected=%f "
                        "actual=%f\n",
                        batch, transpose, precision, iteration, row, column, expected, output[row * n + column]);
                    return false;
                }
            }
    }
    const auto recorded = VulkanArmNx::recordedDispatchCount() - before;
    const bool eligible = requireArmNx && !transpose && precision == BackendConfig::Precision_High;
    MNN_PRINT("MNN mixed graph: batch=%d transpose=%d precision=%d recorded graphs=%llu\n", batch, transpose, precision,
              static_cast<unsigned long long>(recorded));
    return eligible ? recorded > 0 : recorded == 0;
}

static bool quantizedLinear(bool requireArmNx, bool batch, int bits, bool asymmetric,
                            BackendConfig::PrecisionMode precision) {
    const int k = 32, n = 7, blockSize = 8, blocks = k / blockSize;
    BackendConfig config;
    config.precision = precision;
    auto executor =
        Executor::newExecutor(MNN_FORWARD_VULKAN, config, MNN_GPU_TUNING_NONE | (batch ? MNN_GPU_RECORD_BATCH : 0));
    ExecutorScope scope(executor);
    std::vector<int8_t> codes(k * n);
    std::vector<float> alpha(n * blocks * (asymmetric ? 2 : 1)), bias(n);
    const int limit = 1 << (bits - 1);
    for (int i = 0; i < k * n; ++i)
        codes[i] = (i * 7 + i / k) % (2 * limit) - limit;
    for (int i = 0; i < n * blocks; ++i) {
        if (asymmetric) {
            alpha[2 * i] = -0.13f + (i % 3) * 0.02f;
            alpha[2 * i + 1] = 0.003f + (i % 5) * 0.001f;
        } else
            alpha[i] = 0.003f + (i % 5) * 0.001f;
    }
    for (int i = 0; i < n; ++i)
        bias[i] = (i - 3) * 0.03125f;
    if (asymmetric) bias.back() = 10.0f;
    auto encoded =
        IDSTEncoder::encode(nullptr, alpha, blockSize, n * blocks, asymmetric, codes.data(), -limit, {bits, false, 32});
    encoded->quantScale = 1.0f;
    OpT op;
    op.type = OpType_Convolution;
    op.main.type = OpParameter_Convolution2D;
    op.main.value = new Convolution2DT;
    auto conv = op.main.AsConvolution2D();
    conv->common.reset(new Convolution2DCommonT);
    conv->common->inputCount = k;
    conv->common->outputCount = n;
    conv->common->kernelX = conv->common->kernelY = 1;
    conv->common->relu = asymmetric && bits == 4;
    conv->common->relu6 = asymmetric && bits == 8;
    conv->bias = bias;
    conv->quanParameter = std::move(encoded);
    auto input = _Input({1, k, 1, 3}, NCHW);
    input->setName("input");
    auto output = _Convert(Variable::create(Expr::create(&op, {_Convert(input, NC4HW4)})), NCHW);
    output->setName("output");
    const auto data = Variable::save({output});
    Module::Config moduleConfig;
    moduleConfig.shapeMutable = true;
    std::shared_ptr<Module> module(
        Module::load({"input"}, {"output"}, reinterpret_cast<const uint8_t*>(data.data()), data.size(), &moduleConfig),
        Module::destroy);
    if (!module)
        return false;
    std::shared_ptr<Module> clone;
    for (int iteration = 0; iteration < 4; ++iteration) {
        if (iteration == 3) {
            clone.reset(Module::clone(module.get(), true), Module::destroy);
            if (!clone)
                return false;
        }
        const int m = iteration < 2 ? 3 : 1;
        auto x = _Input({1, k, 1, m}, NCHW);
        auto* xv = x->writeMap<float>();
        for (int i = 0; i < m * k; ++i)
            xv[i] = iteration == 1 ? 0.0f : ((i * 11 + iteration) % 23 - 11) / 16.0f;
        auto before = VulkanArmNx::recordedDispatchCount();
        auto y = (iteration < 3 ? module : clone)->onForward({x})[0];
        const auto* actual = y->readMap<float>();
        if (!actual)
            return false;
        if (requireArmNx && iteration != 1 && VulkanArmNx::recordedDispatchCount() <= before)
            return false;
        for (int row = 0; row < m; ++row) {
            for (int channel = 0; channel < n; ++channel) {
                float expected = bias[channel];
                for (int b = 0; b < blocks; ++b) {
                    float maxAbs = 0;
                    for (int j = 0; j < blockSize; ++j)
                        maxAbs = std::max(maxAbs, std::fabs(xv[(b * blockSize + j) * m + row]));
                    const float scale = maxAbs > 0 ? maxAbs / 127.0f : 1.0f;
                    const int index = channel * blocks + b;
                    const float ws = alpha[index * (asymmetric ? 2 : 1) + (asymmetric ? 1 : 0)];
                    const float offset = asymmetric ? alpha[2 * index] + limit * ws : 0.0f;
                    for (int j = 0; j < blockSize; ++j) {
                        const int c = b * blockSize + j;
                        const float a =
                            requireArmNx ? std::nearbyint(xv[c * m + row] / scale) * scale : xv[c * m + row];
                        expected += a * (codes[channel * k + c] * ws + offset);
                    }
                }
                if (asymmetric)
                    expected = std::max(expected, 0.0f);
                if (asymmetric && bits == 8)
                    expected = std::min(expected, 6.0f);
                const float tolerance = precision == BackendConfig::Precision_High ? 0.0002f : 0.005f;
                if (!std::isfinite(actual[channel * m + row]) ||
                    std::fabs(actual[channel * m + row] - expected) > tolerance) {
                    MNN_ERROR("Linear W%d asym=%d batch=%d iter=%d [%d,%d]: expected=%f actual=%f\n", bits, asymmetric,
                              batch, iteration, row, channel, expected, actual[channel * m + row]);
                    return false;
                }
            }
        }
    }
    MNN_PRINT("PASS Linear W%dA8 groups=%d asym=%d batch=%d precision=%d prefill+decode\n", bits, blocks, asymmetric,
              batch, precision);
    return true;
}

int main(int argc, char** argv) {
    bool requireArmNx = false;
    bool fp32Only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-arm-nx") == 0)
            requireArmNx = true;
        else if (std::strcmp(argv[i], "--fp32-only") == 0)
            fp32Only = true;
        else if (std::strcmp(argv[i], "--dump-spirv") == 0 && i + 1 < argc) {
            std::string prefix = argv[++i];
            for (bool int8 : {false, true}) {
                auto code = VulkanArmNx::matMulSpirv(8, 8, 8, int8);
                std::ofstream stream(prefix + (int8 ? "-int8.spv" : "-fp32.spv"), std::ios::binary);
                stream.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(uint32_t));
                if (!stream)
                    return 1;
            }
            return 0;
        } else {
            MNN_ERROR("Usage: %s [--require-arm-nx] [--fp32-only] [--dump-spirv PREFIX]\n", argv[0]);
            return 1;
        }
    }
    if (!InitVulkan())
        return 1;
    auto instance = std::make_shared<VulkanInstance>();
    if (!instance->supportVulkan())
        return 1;
    VulkanDevice device(instance);
    if (!device.success())
        return 1;
    MNN_PRINT("Device: %s, ARM NX path=%d\n", device.proty().deviceName, device.supportsArmNx());
    if (requireArmNx && !device.supportsArmNx())
        return 2;
    if (device.supportsArmNx()) {
        for (const auto& shape : {std::array<uint32_t, 3>{{1, 4, 3}}, {{3, 5, 7}}, {{8, 8, 8}}, {{17, 9, 11}}}) {
            for (bool int8 : {false, true})
                if (!lowLevel(device, shape[0], shape[1], shape[2], int8))
                    return 1;
        }
    } else
        MNN_PRINT("SKIP low-level ARM NX graphs; checking MNN compute fallback\n");
    for (bool batch : {false, true})
        for (bool transpose : {false, true}) {
            for (auto precision : {BackendConfig::Precision_High, BackendConfig::Precision_Low}) {
                if (fp32Only && precision != BackendConfig::Precision_High)
                    continue;
                if (!expression(requireArmNx, batch, transpose, precision))
                    return 1;
            }
        }
    if (requireArmNx)
        for (bool batch : {false, true})
            for (int bits : {4, 8})
                for (bool asymmetric : {false, true}) {
                    for (auto precision : {BackendConfig::Precision_High, BackendConfig::Precision_Low}) {
                        if (fp32Only && precision != BackendConfig::Precision_High)
                            continue;
                        if (!quantizedLinear(requireArmNx, batch, bits, asymmetric, precision))
                            return 1;
                    }
                }
    MNN_PRINT("PASS: ARM NX / MNN mixed execution and fallback\n");
    return 0;
}
