//
//  CoreMLTransformer.cpp
//  MNN
//
//  Created by MNN on 2026/11/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLTransformer.hpp"
#include "CoreMLExecutorWrapper.h"
#include "CoreMLHelper.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/FileLoader.hpp"
#include "core/MNNFileUtils.h"
#include "core/KVCacheImport.hpp"
#include "core/TensorUtils.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#ifdef MNN_COREML_PREFILL_BENCH
#include "MetalPrepare.hpp"
#include <fstream>
#endif
#include <array>
#include <chrono>
#include <cerrno>
#include <thread>
#include <atomic>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include <dispatch/dispatch.h>
#include <climits>
#include <cmath>
#include <future>
#include <sstream>
#include <set>
#include <os/signpost.h>

extern bool isAvailable();

namespace MNN {
namespace {
using Shape = std::vector<int64_t>;
using DType = CoreML__Specification__MILSpec__DataType;
constexpr DType kHalf = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
constexpr DType kInt = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32;
constexpr DType kBool = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL;
struct Value {
    std::string name;
    Shape shape;
};
struct Interval {
    os_log_t log;
    os_signpost_id_t id;
    const bool weight;
    Interval(bool weights, int layer) : weight(weights) {
        static os_log_t logger = os_log_create("org.mnn.coreml", "prefill");
        log = logger;
        id = os_signpost_id_generate(log);
        if (weight) {
            os_signpost_interval_begin(log, id, "CoreMLWeight", "layer=%d", layer);
        } else {
            os_signpost_interval_begin(log, id, "CoreMLLayer", "layer=%d", layer);
        }
    }
    ~Interval() {
        if (weight) {
            os_signpost_interval_end(log, id, "CoreMLWeight");
        } else {
            os_signpost_interval_end(log, id, "CoreMLLayer");
        }
    }
};

// The same channel-first, per-KV-head graph as the validated shared-layer model.
class LayerGraph {
public:
    MlModelBuilder model{9};
    Value input(const std::string& name, Shape shape, bool grayscale = false) {
        info(name, shape, grayscale ? MilTensorInfo::Type::GRAYSCALE_INPUT : MilTensorInfo::Type::INPUT);
        return {name, shape};
    }
    Value operation(const char* type, std::initializer_list<std::pair<std::string, std::string>> args, Shape shape,
                    const std::string& label = "") {
        const auto name = label.empty() ? next() : label;
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder(type));
        op->setName(name);
        for (const auto& a : args) {
            op->addInput(a.first, a.second);
        }
        op->addOutput(name, kHalf, shape);
        model.addOperation(std::move(op));
        return {name, shape};
    }
    template <typename T>
    std::string constant(T value, DType type, Shape shape = {}) {
        const auto name = next();
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, type, shape);
        op->addAttribute("val", MilValueBuilder::createMilValueBuilder(value));
        model.addOperation(std::move(op));
        return name;
    }
    std::string integer(int v) { return constant(v, kInt); }
    std::string real(float v) { return constant(v, kHalf); }
    std::string flag(bool v) { return constant(v, kBool); }
    std::string ints(const std::vector<int>& v) { return constant(v, kInt, {int64_t(v.size())}); }
    Value reshape(Value x, Shape shape) {
        std::vector<int> s(shape.begin(), shape.end());
        return operation("reshape", {{"x", x.name}, {"shape", ints(s)}}, shape);
    }
    Value transpose(Value x, const std::vector<int>& axes) {
        Shape shape;
        for (auto a : axes) {
            shape.push_back(x.shape[a]);
        }
        return operation("transpose", {{"x", x.name}, {"perm", ints(axes)}}, shape);
    }
    Value binary(const char* type, Value x, Value y, Shape shape = {}) {
        return operation(type, {{"x", x.name}, {"y", y.name}}, shape.empty() ? x.shape : shape);
    }
    Value scalar(const char* type, Value x, float y) {
        return operation(type, {{"x", x.name}, {"y", real(y)}}, x.shape);
    }
    Value unary(const char* type, Value x) { return operation(type, {{"x", x.name}}, x.shape); }
    Value cf(Value x, int c, int t) { return reshape(transpose(x, {1, 0}), {1, c, 1, t}); }
    Value norm(Value x, Value gamma, float eps) {
        auto shape = x.shape;
        shape[1] = 1;
        auto square = unary("square", x);
        auto mean =
            operation("reduce_mean", {{"x", square.name}, {"axes", ints({1})}, {"keep_dims", flag(true)}}, shape);
        auto variance = scalar("add", mean, eps);
        auto inv = operation("rsqrt", {{"x", variance.name}, {"epsilon", real(0.0f)}}, shape);
        return binary("mul", binary("mul", x, inv), reshape(gamma, {1, gamma.shape[0], 1, 1}));
    }
    std::vector<Value> split(Value x, int axis, int count) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("split"));
        op->setName(next());
        op->addInput("x", x.name);
        op->addInput("axis", integer(axis));
        op->addInput("num_splits", integer(count));
        Shape shape = x.shape;
        shape[axis] /= count;
        std::vector<Value> result;
        for (int i = 0; i < count; ++i) {
            auto n = next();
            op->addOutput(n, kHalf, shape);
            result.push_back({n, shape});
        }
        model.addOperation(std::move(op));
        return result;
    }
    Value concat(const std::vector<Value>& values, int axis) {
        Shape shape = values[0].shape;
        shape[axis] = 0;
        std::vector<std::string> names;
        for (const auto& v : values) {
            names.push_back(v.name);
            shape[axis] += v.shape[axis];
        }
        auto name = next();
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("concat"));
        op->setName(name);
        op->addInputs("values", names);
        op->addInput("axis", integer(axis));
        op->addInput("interleave", flag(false));
        op->addOutput(name, kHalf, shape);
        model.addOperation(std::move(op));
        return {name, shape};
    }
    Value rope(Value x, Value cosine, Value sine) {
        auto ab = split(x, 1, 2);
        return concat({binary("sub", binary("mul", ab[0], cosine), binary("mul", ab[1], sine)),
                       binary("add", binary("mul", ab[1], cosine), binary("mul", ab[0], sine))},
                      1);
    }
    Value matmul(Value x, Value y, Shape shape, bool ty = false) {
        return operation(
            "matmul", {{"x", x.name}, {"y", y.name}, {"transpose_x", flag(false)}, {"transpose_y", flag(ty)}}, shape);
    }
    Value project(Value x, Value w, int t) {
        auto flat = transpose(reshape(x, {w.shape[1], t}), {1, 0});
        return cf(matmul(flat, w, {t, w.shape[0]}, true), int(w.shape[0]), t);
    }
    void output(Value v, const std::string& name) {
        operation("identity", {{"x", v.name}}, v.shape, name);
        info(name, v.shape, MilTensorInfo::Type::OUTPUT);
    }

private:
    int mId = 0;
    std::string next() { return "v" + std::to_string(mId++); }
    void info(const std::string& name, Shape shape, MilTensorInfo::Type type) {
        std::unique_ptr<MilTensorInfo> p(new MilTensorInfo);
        p->name = name;
        p->shape = shape;
        p->data_type = kHalf;
        p->type = type;
        if (type != MilTensorInfo::Type::OUTPUT) {
            model.addInput(std::move(p));
        } else {
            model.addOutput(std::move(p));
        }
    }
};

// Parallel row slices share this descriptor and keep independent, reusable input buffers.
struct WeightFile {
    file_t fd = INVALID_FILE;
    size_t size = 0;
    void open(const std::string& path) {
        fd = MNNOpenFile(path.c_str(), MNN_FILE_READ);
        if (fd != INVALID_FILE) {
            size = MNNGetFileSize(fd);
        }
    }
    ~WeightFile() {
        if (fd != INVALID_FILE) {
            MNNCloseFile(fd);
        }
    }
};

struct Projection {
    int rows = 0, cols = 0;
    std::vector<uint8_t> serialized;
    std::shared_ptr<ConvolutionCommon::Int8Common> quant;
    file_t fd = INVALID_FILE;
    size_t rawOffset = 0;
#ifdef MNN_COREML_PREFILL_BENCH
    std::shared_ptr<BenchMappedWeight> gpuSource;
#endif

    void bindExternalQ4(const Convolution2D* conv, const WeightFile& file) {
        const auto q = conv->quanParameter();
        const auto ext = conv->external();
        const size_t count = size_t(rows) * cols;
        // Other encodings can require a codebook or a bias/scale adjustment in the IDST decoder.
        if (file.fd == INVALID_FILE || file.size == INVALID_SIZE || !q->readType() || q->aMin() <= 0 || q->quantScale() != 1.0f ||
            q->scaleStorage() != ScaleStorageType_FP16 || ext->Get(2) != count / 16) {
            return;
        }
        const size_t offset = ext->Get(0), bytes = ext->Get(1);
        if (offset >= file.size || bytes > file.size - offset || bytes <= count / 2 ||
            count / 16 > file.size - offset - bytes) {
            return;
        }
        uint8_t header[34];
        const size_t headerSize = bytes - count / 2;
        if (headerSize > sizeof(header) || pread(file.fd, header, headerSize, offset) != ssize_t(headerSize)) {
            return;
        }
        const size_t rank = header[0], width = q->shapeInt32() ? 4 : 2;
        if (rank == 0 || rank > 4 || headerSize != 1 + rank * width + 1 + 16 ||
            header[1 + rank * width] != 16) {
            return;
        }
        size_t elements = 1;
        for (size_t axis = 0; axis < rank; ++axis) {
            uint32_t dim = 0;
            memcpy(&dim, header + 1 + axis * width, width);
            if (dim == 0 || elements > count / dim) {
                return;
            }
            elements *= dim;
        }
        const auto codebook = reinterpret_cast<const int8_t*>(header + 2 + rank * width);
        for (int i = 0; i < 16; ++i) {
            if (codebook[i] != i - 8) {
                return;
            }
        }
        if (elements == count) {
            fd = file.fd;
            rawOffset = offset + bytes - count / 2;
        }
    }

    bool load(const Convolution2D* conv, const std::string& path, bool channelInt8, const WeightFile& file) {
        if (!conv || !conv->common() || !conv->quanParameter()) {
            return false;
        }
        auto c = conv->common();
        auto q = conv->quanParameter();
        rows = c->outputCount();
        cols = c->inputCount();
        if (rows <= 0 || cols <= 0 || int64_t(rows) * cols > INT_MAX || c->kernelX() != 1 || c->kernelY() != 1 ||
            c->group() != 1 || c->strideX() != 1 || c->strideY() != 1 || c->padX() != 0 || c->padY() != 0 ||
            c->relu() || c->relu6() || q->type() != 1 ||
            q->index() || q->has_scaleInt() ||
            (conv->external() && conv->external()->size() >= 4 && conv->external()->Get(3) != 0)) {
            return false;
        }
        if (conv->bias()) {
            for (float b : *conv->bias()) {
                if (b != 0.0f) {
                    return false;
                }
            }
        }
        if (channelInt8 && (cols % 64 || q->buffer() || !conv->external() || conv->external()->size() < 3)) {
            return false;
        }
        OpT op;
        op.type = OpType_Convolution;
        op.main.type = OpParameter_Convolution2D;
        op.main.value = conv->UnPack();
        op.externalPath = path;
        flatbuffers::FlatBufferBuilder fb;
        fb.Finish(Op::Pack(fb, &op));
        serialized.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
        if (channelInt8) {
            // Keep only the external weight descriptor alongside the native decode model.
            const auto ext = conv->external();
            const auto alphaBytes = q->scaleStorage() == ScaleStorageType_FP16 ? 2 : 4;
            const auto groups = int64_t(rows) * cols / 64;
            const bool scaleSizeValid = ext->Get(2) == groups * alphaBytes || ext->Get(2) == 2 * groups * alphaBytes;
            if (ext->Get(0) < 0 || ext->Get(1) <= 0 || !scaleSizeValid) {
                return false;
            }
            bindExternalQ4(conv, file);
            return true;
        }
        quant = readQuant();
        if (!quant || !quant->weight.get() || !quant->getAlphaFloat() ||
            (!quant->canUseInt4 && quant->originBits != 8)) {
            return false;
        }
        quant->alphaHalf.release();
        const size_t count = size_t(rows) * cols;
        const int groups = quant->alphaSize / (quant->asymmetric ? 2 : 1);
        return groups > 0 && count % groups == 0 &&
               quant->weight.size() == (quant->canUseInt4 ? (count + 1) / 2 : count);
    }
    std::shared_ptr<ConvolutionCommon::Int8Common> readQuant() const {
        return ConvolutionCommon::load(flatbuffers::GetRoot<Op>(serialized.data()), nullptr, false, true, nullptr,
                                       true);
    }
    // Keep both image axes within the device limits for wide Qwen projections.
    int imageWidth() const { return size_t(rows) * cols % 4096 == 0 ? 4096 : cols; }
    int imageHeight() const { return size_t(rows) * cols / imageWidth(); }
    static bool convertRow(const uint8_t* packed, const float* alpha, bool asymmetric, int cols, uint8_t* out,
                           int16_t* channelScale, float* scratch) {
        float peak = 0.0f;
        const int stride = asymmetric ? 2 : 1;
        for (int k = 0; k < cols; k += 64) {
            const auto group = alpha + k / 64 * stride;
            const float scale = group[stride - 1], bias = asymmetric ? group[0] : 0.0f;
            if (!std::isfinite(scale) || !std::isfinite(bias)) {
                return false;
            }
#ifdef __aarch64__
            const auto a = vld1q_u8(packed + k / 2), b = vld1q_u8(packed + k / 2 + 16);
            const auto mask = vdupq_n_u8(15);
            auto lo = vminq_u8(vandq_u8(a, mask), vandq_u8(b, mask));
            auto hi = vminq_u8(vshrq_n_u8(a, 4), vshrq_n_u8(b, 4));
            int minCode = int(vminvq_u8(vminq_u8(lo, hi))) - 8;
            lo = vmaxq_u8(vandq_u8(a, mask), vandq_u8(b, mask));
            hi = vmaxq_u8(vshrq_n_u8(a, 4), vshrq_n_u8(b, 4));
            int maxCode = int(vmaxvq_u8(vmaxq_u8(lo, hi))) - 8;
            peak = std::max(peak, std::max(std::abs(minCode * scale + bias), std::abs(maxCode * scale + bias)));
#else
            dequantizeCoreMLWeightBlock(reinterpret_cast<const int8_t*>(packed), scratch + k, k, 64, true, scale, bias);
            for (int i = 0; i < 64; ++i) {
                peak = std::max(peak, std::abs(scratch[k + i]));
            }
#endif
        }
        float step = peak == 0.0f ? 1.0f : std::max(peak / 127.0f, 5.96046448e-8f);
        MNNQuantizeFP16(&step, channelScale, 1);
        MNNDequantizeFP16(channelScale, &step, 1);
        if (!(step > 0 && std::isfinite(step))) {
            return false;
        }
#ifdef __aarch64__
        const float32x4_t first = {-8, -7, -6, -5};
        for (int k = 0; k < cols; k += 64) {
            const auto group = alpha + k / 64 * stride;
            const auto bias = vdupq_n_f32(asymmetric ? group[0] : 0.0f);
            auto four = [&](int offset) {
                auto weights = vfmaq_n_f32(bias, vaddq_f32(first, vdupq_n_f32(float(offset))), group[stride - 1]);
                return vqmovn_s32(vcvtnq_s32_f32(vdivq_f32(weights, vdupq_n_f32(step))));
            };
            const auto low = vqmovn_s16(vcombine_s16(four(0), four(4)));
            const auto high = vqmovn_s16(vcombine_s16(four(8), four(12)));
            auto table = vreinterpretq_u8_s8(vmaxq_s8(vcombine_s8(low, high), vdupq_n_s8(-127)));
            // Grayscale storage carries signed W8 codes with an unsigned offset of 128.
            table = veorq_u8(table, vdupq_n_u8(128));
            for (int j = 0; j < 32; j += 16) {
                auto bytes = vld1q_u8(packed + k / 2 + j);
                uint8x16x2_t decoded = {vqtbl1q_u8(table, vshrq_n_u8(bytes, 4)),
                                        vqtbl1q_u8(table, vandq_u8(bytes, vdupq_n_u8(15)))};
                vst2q_u8(out + k + 2 * j, decoded);
            }
        }
#else
        for (int k = 0; k < cols; ++k) {
            out[k] = uint8_t(128 + int(std::nearbyint(std::max(-127.0f, std::min(127.0f, scratch[k] / step)))));
        }
#endif
        return true;
    }
    bool toChannelInt8(CVPixelBufferRef image, int16_t* scales, std::array<std::vector<uint8_t>, 6>& scratchWeight) const {
        std::shared_ptr<ConvolutionCommon::Int8Common> source;
        const uint8_t* packed = nullptr;
        const float* alpha = nullptr;
        bool asymmetric = true;
        if (fd == INVALID_FILE) {
            source = readQuant();
            const size_t count = size_t(rows) * cols;
            if (!source || !source->canUseInt4 || source->weight.size() != count / 2 ||
                source->alphaSize != count / 64 * (source->asymmetric ? 2 : 1) || !source->weight.get()) {
                return false;
            }
            alpha = source->getAlphaFloat();
            if (!alpha) {
                return false;
            }
            packed = reinterpret_cast<const uint8_t*>(source->weight.get());
            asymmetric = source->asymmetric;
        }
        const size_t slices = std::min(rows, fd != INVALID_FILE ? 6 : 4);
        std::atomic<bool> valid{true};
        if (CVPixelBufferLockBaseAddress(image, 0) != kCVReturnSuccess) {
            return false;
        }
        const auto width = CVPixelBufferGetWidth(image);
        const auto pitch = CVPixelBufferGetBytesPerRow(image);
        auto* codes = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(image));
        if (!codes || pitch < width) {
            CVPixelBufferUnlockBaseAddress(image, 0);
            return false;
        }
        auto convertRows = [&](size_t slice) {
            std::vector<uint8_t> staging(pitch != width ? cols : 0);
            const size_t firstRow = slice * rows / slices;
            const size_t endRow = (slice + 1) * rows / slices;
            const auto* sliceCodes = packed;
            const int16_t* sliceAlpha = nullptr;
            if (fd != INVALID_FILE) {
                const auto elements = (endRow - firstRow) * cols;
                auto& bytes = scratchWeight[slice];
                const size_t packedBytes = elements / 2, alphaBytes = elements / 16;
                if (bytes.size() < packedBytes + alphaBytes) {
                    bytes.resize(packedBytes + alphaBytes);
                }
                auto readRange = [&](uint8_t* target, size_t length, size_t offset) {
                    while (length > 0) {
                        const auto n = pread(fd, target, length, offset);
                        if (n < 0 && errno == EINTR) {
                            continue;
                        }
                        if (n <= 0) {
                            return false;
                        }
                        target += n;
                        length -= n;
                        offset += n;
                    }
                    return true;
                };
                if (!readRange(bytes.data(), packedBytes, rawOffset + firstRow * cols / 2) ||
                    !readRange(bytes.data() + packedBytes, alphaBytes,
                               rawOffset + size_t(rows) * cols / 2 + firstRow * cols / 16)) {
                    valid.store(false, std::memory_order_relaxed);
                    return;
                }
                sliceCodes = bytes.data();
                sliceAlpha = reinterpret_cast<const int16_t*>(bytes.data() + packedBytes);
            } else {
                sliceCodes += firstRow * cols / 2;
            }
            std::vector<float> rowAlpha(sliceAlpha ? cols / 32 : 0);
            std::vector<float> scratch;
#ifndef __aarch64__
            scratch.resize(cols);
#endif
            for (size_t r = firstRow; r < endRow; ++r) {
                size_t first = r * cols;
                const float* groups;
                if (sliceAlpha) {
                    MNNDequantizeFP16(sliceAlpha + (r - firstRow) * cols / 32, rowAlpha.data(), rowAlpha.size());
                    groups = rowAlpha.data();
                } else {
                    groups = alpha + first / 64 * (asymmetric ? 2 : 1);
                }
                auto* dst = staging.empty() ? codes + first : staging.data();
                if (!convertRow(sliceCodes + (r - firstRow) * cols / 2, groups, asymmetric, cols, dst, scales + r, scratch.data())) {
                    valid.store(false, std::memory_order_relaxed);
                    return;
                }
                if (!staging.empty()) {
                    size_t offset = first;
                    for (size_t k = 0; k < size_t(cols);) {
                        const auto n = std::min(size_t(cols) - k, width - offset % width);
                        memcpy(codes + offset / width * pitch + offset % width, dst + k, n);
                        offset += n;
                        k += n;
                    }
                }
            }
        };
        dispatch_apply_f(slices, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &convertRows,
                         [](void* job, size_t slice) { (*static_cast<decltype(convertRows)*>(job))(slice); });
        return CVPixelBufferUnlockBaseAddress(image, 0) == kCVReturnSuccess && valid.load(std::memory_order_relaxed);
    }
    void decode(int16_t* out) const {
        const size_t count = size_t(rows) * cols;
        const int groups = quant->alphaSize / (quant->asymmetric ? 2 : 1);
        const size_t block = count / groups;
        const auto alpha = quant->getAlphaFloat();
        float scratch[1024];
        for (int g = 0; g < groups; ++g) {
            float scale = alpha[quant->asymmetric ? 2 * g + 1 : g];
            float bias = quant->asymmetric ? alpha[2 * g] : 0.0f;
            for (size_t j = 0; j < block; j += 1024) {
                auto n = std::min<size_t>(1024, block - j);
                auto start = size_t(g) * block + j;
                dequantizeCoreMLWeightBlock(quant->weight.get(), scratch, start, n, quant->canUseInt4, scale, bias);
                MNNQuantizeFP16(scratch, out + start, n);
            }
        }
    }
};

struct Layer {
    std::string attention;
    int heads = 0, kv = 0, dim = 0, hidden = 0, intermediate = 0;
    std::array<Projection, 7> weights;
    std::array<std::shared_ptr<Tensor>, 4> norms;
    std::array<float, 4> eps;
};
const char* weightName(int i) {
    const char* names[] = {"wq", "wk", "wv", "wo", "wg", "wu", "wd"};
    return names[i];
}
const char* normName(int i) {
    const char* names[] = {"an", "qn", "kn", "fn"};
    return names[i];
}
std::shared_ptr<Tensor> halfTensor(const std::vector<int>& shape) {
    auto t =
        std::shared_ptr<Tensor>(Tensor::create(shape, halide_type_t(halide_type_float, 16), nullptr, Tensor::CAFFE));
    return t && t->host<void>() ? t : nullptr;
}
std::shared_ptr<Tensor> normTensor(const LayerNorm* norm, int size, const std::string& path) {
    if (!norm || !norm->useRMSNorm() || norm->group() != 1 || !norm->axis() || norm->axis()->size() != 1 ||
        norm->axis()->Get(0) != -1 || !std::isfinite(norm->epsilon()) || norm->epsilon() <= 0) {
        return nullptr;
    }
    std::vector<float> gamma(size), beta;
    if (norm->external() && norm->external()->size() >= 3) {
        auto ext = norm->external();
        if (ext->Get(1) != size * sizeof(float) || ext->Get(2) < 0 || ext->Get(2) > size * sizeof(float)) {
            return nullptr;
        }
        FileLoader reader(path.c_str(), true);
        beta.resize(ext->Get(2) / sizeof(float));
        if (!reader.valid() || reader.offset(ext->Get(0)) != 0 ||
            !reader.read(reinterpret_cast<char*>(gamma.data()), size * sizeof(float)) ||
            (!beta.empty() && !reader.read(reinterpret_cast<char*>(beta.data()), beta.size() * sizeof(float)))) {
            return nullptr;
        }
    } else if (norm->gamma() && norm->gamma()->size() == size) {
        std::copy(norm->gamma()->begin(), norm->gamma()->end(), gamma.begin());
        if (norm->beta()) {
            beta.assign(norm->beta()->begin(), norm->beta()->end());
        }
    } else {
        return nullptr;
    }
    for (float b : beta) {
        if (b != 0) {
            return nullptr;
        }
    }
    auto t = halfTensor({size});
    if (t) {
        MNNQuantizeFP16(gamma.data(), t->host<int16_t>(), size);
    }
    return t;
}
void buildLayer(LayerGraph& g, const Layer& l, int t, bool channelInt8) {
    auto x = g.input("x", {t, l.hidden});
    auto cosine = g.cf(g.input("cosine", {t, l.dim / 2}), l.dim / 2, t);
    auto sine = g.cf(g.input("sine", {t, l.dim / 2}), l.dim / 2, t);
    std::vector<Value> w, norms;
    for (int i = 0; i < 7; ++i) {
        const std::string name = weightName(i);
        Shape shape{l.weights[i].rows, l.weights[i].cols};
        if (channelInt8) {
            const auto& projection = l.weights[i];
            auto bytes = g.input(name, {1, 1, projection.imageHeight(), projection.imageWidth()}, true);
            auto signedCodes = g.scalar("sub", g.reshape(bytes, shape), 128.0f);
            w.push_back(g.binary("mul", signedCodes, g.input(name + "_scale", {shape[0], 1})));
        } else {
            w.push_back(g.input(name, shape));
        }
    }
    for (int i = 0; i < 4; ++i) {
        norms.push_back(g.input(normName(i), {i == 1 || i == 2 ? l.dim : l.hidden}));
    }
    x = g.cf(x, l.hidden, t);
    auto normalized = g.norm(x, norms[0], l.eps[0]);
    auto q = g.reshape(g.project(normalized, w[0], t), {l.heads, l.dim, 1, t});
    auto k = g.reshape(g.project(normalized, w[1], t), {l.kv, l.dim, 1, t});
    auto v = g.reshape(g.project(normalized, w[2], t), {l.kv, l.dim, 1, t});
    q = g.rope(g.norm(q, norms[1], l.eps[1]), cosine, sine);
    k = g.rope(g.norm(k, norms[2], l.eps[2]), cosine, sine);
    auto qs = g.split(q, 0, l.kv), ks = g.split(k, 0, l.kv), vs = g.split(v, 0, l.kv);
    std::vector<float> mask(size_t(t) * t, 0.0f);
    for (int row = 0; row < t; ++row) {
        std::fill(mask.begin() + size_t(row) * t + row + 1, mask.begin() + size_t(row + 1) * t, -10000.0f);
    }
    Value maskValue{g.constant(mask, kHalf, {int64_t(mask.size())}), {int64_t(mask.size())}};
    maskValue = g.reshape(maskValue, {t, t});
    std::vector<Value> attention;
    int group = l.heads / l.kv;
    for (int i = 0; i < l.kv; ++i) {
        auto qi = g.transpose(g.reshape(qs[i], {group, l.dim, t}), {0, 2, 1});
        auto ki = g.transpose(g.reshape(ks[i], {1, l.dim, t}), {0, 2, 1});
        auto vi = g.transpose(g.reshape(vs[i], {1, l.dim, t}), {0, 2, 1});
        auto scores = g.matmul(qi, ki, {group, t, t}, true);
        scores = g.binary("add", g.scalar("mul", scores, 1.0f / std::sqrt(float(l.dim))), maskValue);
        auto prob = g.operation("softmax", {{"x", scores.name}, {"axis", g.integer(-1)}}, scores.shape);
        auto h = g.matmul(prob, vi, {group, t, l.dim});
        attention.push_back(g.reshape(g.transpose(h, {0, 2, 1}), {1, group * l.dim, 1, t}));
    }
    auto residual = g.binary("add", x, g.project(g.concat(attention, 1), w[3], t));
    auto z = g.norm(residual, norms[3], l.eps[3]);
    auto gate = g.project(z, w[4], t), up = g.project(z, w[5], t);
    auto sigmoid = g.scalar("mul", g.scalar("add", g.unary("tanh", g.scalar("mul", gate, 0.5f)), 1.0f), 0.5f);
    auto mixed = g.binary("mul", gate, g.binary("mul", up, sigmoid));
    auto y = g.binary("add", residual, g.project(mixed, w[6], t));
    g.output(g.transpose(g.reshape(y, {l.hidden, t}), {1, 0}), "hidden");
    g.output(g.transpose(g.reshape(k, {l.kv, l.dim, t}), {2, 0, 1}), "key");
    g.output(g.transpose(g.reshape(v, {l.kv, l.dim, t}), {2, 0, 1}), "value");
}

std::shared_ptr<CoreMLExecutorWrapper> compileLayer(const Layer& l, int t, bool channelInt8) {
    LayerGraph g;
    buildLayer(g, l, t, channelInt8);
    std::shared_ptr<CoreMLExecutorWrapper> program(new CoreMLExecutorWrapper(0));
    if (!program->compileModel(g.model.build(), "", nullptr, true)) {
        return nullptr;
    }
    return program;
}
} // namespace

struct CoreMLTransformer::Impl {
    WeightFile weightFile;
    std::array<std::vector<uint8_t>, 6> scratchWeight;
    std::vector<Layer> layers;
    std::vector<std::string> rotary;
    std::string head;
    std::map<std::string, std::shared_ptr<CoreMLExecutorWrapper>> programs;
    struct Slot {
        std::array<std::shared_ptr<Tensor>, 7> weights, scales;
        std::array<CoreMLWeightImage, 7> images;
    };
#ifdef MNN_COREML_PREFILL_BENCH
    std::unique_ptr<BenchMetalPrepare> gpu;
    std::string fixedDirectory;
    bool exportLayers(const std::string& directory, const std::string& source) const {
        std::ofstream path(directory + "/source.txt");
        std::ofstream layout(directory + "/projections.tsv");
        path << source;
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            LayerGraph g;
            buildLayer(g, layer, 128, false);
            auto* spec = g.model.build();
            std::vector<uint8_t> bytes(core_ml__specification__model__get_packed_size(spec));
            core_ml__specification__model__pack(spec, bytes.data());
            std::ofstream model(directory + "/" + std::to_string(i) + ".mlmodel", std::ios::binary);
            model.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            model.close();
            if (!model) {
                return false;
            }
            for (int j = 0; j < 7; ++j) {
                const auto& w = layer.weights[j];
                if (w.fd == INVALID_FILE) {
                    return false;
                }
                layout << i << '\t' << weightName(j) << '\t' << w.rows << '\t' << w.cols << '\t' << w.rawOffset << '\n';
            }
        }
        layout.close();
        path.close();
        return layout.good() && path.good();
    }
#endif
    bool channelInt8 = false;
    std::array<Slot, 2> slots;
    std::mutex mutex;
    bool prepare(int index, Slot& slot) {
        Interval interval(true, index);
#ifdef MNN_COREML_PREFILL_BENCH
        std::array<BenchProjection, 7> batch;
#endif
        for (int i = 0; i < 7; ++i) {
            const auto& w = layers[index].weights[i];
            if (channelInt8) {
                auto& image = slot.images[i];
                if (!image || CVPixelBufferGetWidth(image.get()) != size_t(w.imageWidth()) ||
                    CVPixelBufferGetHeight(image.get()) != size_t(w.imageHeight())) {
                    image = createCoreMLWeightImage(w.imageWidth(), w.imageHeight());
                }
                if (!slot.scales[i] || slot.scales[i]->length(0) != w.rows) {
                    slot.scales[i] = halfTensor({w.rows, 1});
                }
#ifdef MNN_COREML_PREFILL_BENCH
                if (gpu && image && slot.scales[i]) {
                    batch[i] = {w.gpuSource.get(), w.rows, w.cols, image.get(), slot.scales[i]->host<int16_t>()};
                    continue;
                }
#endif
                if (!image || !slot.scales[i] || !w.toChannelInt8(image.get(), slot.scales[i]->host<int16_t>(), scratchWeight)) {
                    return false;
                }
            } else {
                auto& t = slot.weights[i];
                if (!t || t->length(0) != w.rows || t->length(1) != w.cols) {
                    t = halfTensor({w.rows, w.cols});
                }
                if (!t) {
                    return false;
                }
                w.decode(t->host<int16_t>());
            }
        }
#ifdef MNN_COREML_PREFILL_BENCH
        if (gpu) {
            return gpu->convert(batch, index);
        }
#endif
        return true;
    }
};
CoreMLTransformer::CoreMLTransformer() : mImpl(new Impl) {}
CoreMLTransformer::~CoreMLTransformer() = default;
const std::vector<std::string>& CoreMLTransformer::rotaryInputs() const {
    return mImpl->rotary;
}
const std::string& CoreMLTransformer::headInput() const {
    return mImpl->head;
}
int CoreMLTransformer::hiddenSize() const {
    return mImpl->layers[0].hidden;
}
bool CoreMLTransformer::supportsAttentionScale(float scale) const {
    for (const auto& layer : mImpl->layers) {
        if (scale != 0 && std::abs(scale - 1.0f / std::sqrt(float(layer.dim))) > 1e-7f) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<CoreMLTransformer> CoreMLTransformer::create(const Net* net, const std::string& path) {
    return create(net, path, false);
}
std::shared_ptr<CoreMLTransformer> CoreMLTransformer::create(const Net* net, const std::string& path,
                                                             bool channelInt8) {
    if (!::isAvailable() || !net || !net->oplists() || !net->tensorName()) {
        return nullptr;
    }
    std::shared_ptr<CoreMLTransformer> result(new CoreMLTransformer);
    auto& impl = *result->mImpl;
    impl.channelInt8 = channelInt8;
#ifdef MNN_COREML_PREFILL_BENCH
    if (channelInt8 && getenv("MNN_PREFILL_BENCH_GPU")) {
        impl.gpu.reset(new BenchMetalPrepare);
    }
    if (channelInt8 && getenv("MNN_PREFILL_BENCH_FIXED")) {
        impl.fixedDirectory = getenv("MNN_PREFILL_BENCH_FIXED");
    }
#endif
    if (channelInt8) {
        impl.weightFile.open(path);
    }
    std::map<int, const Op*> producer;
    std::map<int, std::vector<const Op*>> consumers;
    for (auto op : *net->oplists()) {
        if (op->outputIndexes()) {
            for (int index : *op->outputIndexes()) {
                if (index < 0 || index >= net->tensorName()->size() || producer.count(index)) {
                    return nullptr;
                }
                producer[index] = op;
            }
        }
        if (op->inputIndexes()) {
            for (int index : *op->inputIndexes()) {
                if (index < 0 || index >= net->tensorName()->size()) {
                    return nullptr;
                }
                consumers[index].push_back(op);
            }
        }
    }
    auto get = [&](int index, OpType type) -> const Op* {
        auto p = producer.find(index);
        return p != producer.end() && p->second->type() == type ? p->second : nullptr;
    };
    auto sole = [&](int index, OpType type) -> const Op* {
        auto p = consumers.find(index);
        return p != consumers.end() && p->second.size() == 1 && p->second[0]->type() == type ? p->second[0] : nullptr;
    };
    auto isInput = [&](int index, const char* name) {
        auto input = get(index, OpType_Input);
        return input && net->tensorName()->Get(index)->str() == name;
    };
    auto hiddenInput = [&](int index) {
        // These operations preserve the linear token/channel order supplied by Llm::embedding.
        for (int depth = 0; depth < 8; ++depth) {
            if (isInput(index, "input_ids")) {
                return true;
            }
            auto p = producer.find(index);
            if (p == producer.end()) {
                return false;
            }
            auto op = p->second;
            if ((op->type() != OpType_Reshape && op->type() != OpType_ConvertTensor) || !op->inputIndexes() ||
                op->inputIndexes()->size() < 1) {
                return false;
            }
            if (op->type() == OpType_ConvertTensor) {
                auto cvt = op->main_as_TensorConvertInfo();
                if (!cvt || cvt->source() != MNN_DATA_FORMAT_NCHW || cvt->dest() != MNN_DATA_FORMAT_NC4HW4) {
                    return false;
                }
            }
            index = op->inputIndexes()->Get(0);
        }
        return false;
    };
    std::set<std::string> attentionNames;
    const Op* lastFF = nullptr;
    const Op* lastDown = nullptr;
    for (auto op : *net->oplists()) {
        if (op->type() == OpType_LinearAttention) {
            return nullptr;
        }
        if (op->type() != OpType_Attention) {
            continue;
        }
        auto ap = op->main_as_AttentionParam();
        if (!ap || !ap->kv_cache() || ap->kv_shared_layer_index() >= 0 ||
            (ap->kv_shared_layer() && ap->kv_shared_layer()->size()) ||
            (ap->mhq_quant() && ap->mhq_quant()->size()) || !ap->output_c4() || !op->name() || !op->inputIndexes() ||
            op->inputIndexes()->size() != 4 || op->outputIndexes()->size() != 1) {
            return nullptr;
        }
        auto rope = get(op->inputIndexes()->Get(0), OpType_RoPE);
        if (!rope || rope != get(op->inputIndexes()->Get(1), OpType_RoPE) || rope->inputIndexes()->size() != 4) {
            return nullptr;
        }
        auto rp = rope->main_as_RoPEParam();
        if (!rp || !isInput(op->inputIndexes()->Get(3), "attention_mask") ||
            !attentionNames.insert(op->name()->str()).second) {
            return nullptr;
        }
        auto qkv = get(rope->inputIndexes()->Get(0), OpType_FusedLinear);
        if (!qkv || qkv != get(rope->inputIndexes()->Get(1), OpType_FusedLinear) ||
            qkv != get(op->inputIndexes()->Get(2), OpType_FusedLinear)) {
            return nullptr;
        }
        auto qp = qkv->main_as_FusedLinearParam();
        if (!qp || qp->act_silu_mul() || !qp->convs() || qp->convs()->size() != 3 || !qkv->outputIndexes() ||
            qkv->outputIndexes()->size() != (qp->has_ln() ? 4 : 3)) {
            return nullptr;
        }
        auto output = sole(op->outputIndexes()->Get(0), OpType_Convolution);
        if (!output || output->outputIndexes()->size() != 1) {
            return nullptr;
        }
        auto ff = sole(output->outputIndexes()->Get(0), OpType_FusedLinear);
        if (!ff || ff->inputIndexes()->size() != 2 || ff->outputIndexes()->size() != 2) {
            return nullptr;
        }
        auto fp = ff->main_as_FusedLinearParam();
        if (!fp || !fp->has_ln() || !fp->act_silu_mul() || !fp->convs() || fp->convs()->size() != 2) {
            return nullptr;
        }
        auto down = sole(ff->outputIndexes()->Get(0), OpType_Convolution);
        if (!down || down->outputIndexes()->size() != 1) {
            return nullptr;
        }
        const LayerNorm* an = nullptr;
        if (!lastFF) {
            if (qp->has_ln() || qkv->inputIndexes()->size() != 1) {
                return nullptr;
            }
            auto ln = get(qkv->inputIndexes()->Get(0), OpType_LayerNorm);
            if (!ln || ln->inputIndexes()->size() != 1 || ff->inputIndexes()->Get(0) != ln->inputIndexes()->Get(0) ||
                !hiddenInput(ln->inputIndexes()->Get(0))) {
                return nullptr;
            }
            an = ln->main_as_LayerNorm();
        } else {
            if (!qp->has_ln() || qkv->inputIndexes()->size() != 2 || qkv->outputIndexes()->size() != 4 ||
                qkv->inputIndexes()->Get(0) != lastFF->outputIndexes()->Get(1) ||
                qkv->inputIndexes()->Get(1) != lastDown->outputIndexes()->Get(0) ||
                ff->inputIndexes()->Get(0) != qkv->outputIndexes()->Get(3)) {
                return nullptr;
            }
            an = qp->ln();
        }
        if (ff->inputIndexes()->Get(1) != output->outputIndexes()->Get(0) ||
            rope->inputIndexes()->Get(0) != qkv->outputIndexes()->Get(0) ||
            rope->inputIndexes()->Get(1) != qkv->outputIndexes()->Get(1) ||
            op->inputIndexes()->Get(2) != qkv->outputIndexes()->Get(2)) {
            return nullptr;
        }
        Layer l;
        l.attention = op->name()->str();
        l.heads = rp->num_head();
        l.kv = rp->kv_num_head();
        l.dim = rp->head_dim();
        if (l.heads <= 0 || l.kv <= 0 || l.dim <= 0 || l.dim % 2 || l.heads % l.kv ||
            rp->rope_cut_head_dim() != l.dim) {
            return nullptr;
        }
        const Convolution2D* convs[] = {
            qp->convs()->Get(0), qp->convs()->Get(1), qp->convs()->Get(2),          output->main_as_Convolution2D(),
            fp->convs()->Get(0), fp->convs()->Get(1), down->main_as_Convolution2D()};
        for (int i = 0; i < 7; ++i) {
            if (!l.weights[i].load(convs[i], path, channelInt8, impl.weightFile)) {
                return nullptr;
            }
        }
#ifdef MNN_COREML_PREFILL_BENCH
        if (impl.gpu) {
            for (auto& w : l.weights) {
                w.gpuSource = BenchMappedWeight::map(w.fd, w.rawOffset, size_t(w.rows) * w.cols);
                if (!w.gpuSource) {
                    return nullptr;
                }
            }
        }
#endif
        l.hidden = l.weights[0].cols;
        l.intermediate = l.weights[4].rows;
        const int rows[] = {l.heads * l.dim, l.kv * l.dim,   l.kv * l.dim, l.hidden,
                            l.intermediate,  l.intermediate, l.hidden};
        const int cols[] = {l.hidden, l.hidden, l.hidden, l.heads * l.dim, l.hidden, l.hidden, l.intermediate};
        for (int i = 0; i < 7; ++i) {
            if (l.weights[i].rows != rows[i] || l.weights[i].cols != cols[i]) {
                return nullptr;
            }
        }
        const LayerNorm* norms[] = {an, rp->q_norm(), rp->k_norm(), fp->ln()};
        for (int i = 0; i < 4; ++i) {
            l.norms[i] = normTensor(norms[i], i == 1 || i == 2 ? l.dim : l.hidden, path);
            if (!l.norms[i]) {
                return nullptr;
            }
            l.eps[i] = norms[i]->epsilon();
        }
        std::vector<std::string> rotary;
        for (int i = 2; i < 4; ++i) {
            rotary.push_back(net->tensorName()->Get(rope->inputIndexes()->Get(i))->str());
        }
        if (impl.rotary.empty()) {
            impl.rotary = rotary;
        }
        if (impl.rotary != rotary ||
            (!impl.layers.empty() && (impl.layers[0].hidden != l.hidden || impl.layers[0].dim != l.dim))) {
            return nullptr;
        }
        impl.layers.push_back(std::move(l));
        lastFF = ff;
        lastDown = down;
    }
    if (impl.layers.empty()) {
        return nullptr;
    }
    auto residual = sole(lastDown->outputIndexes()->Get(0), OpType_BinaryOp);
    if (!residual || residual->main_as_BinaryOp()->opType() != BinaryOpOperation_ADD ||
        residual->main_as_BinaryOp()->activationType() != 0 || residual->inputIndexes()->size() != 2 ||
        residual->inputIndexes()->Get(0) != lastFF->outputIndexes()->Get(1) ||
        residual->inputIndexes()->Get(1) != lastDown->outputIndexes()->Get(0)) {
        return nullptr;
    }
    impl.head = net->tensorName()->Get(residual->outputIndexes()->Get(0))->str();
#ifdef MNN_COREML_PREFILL_BENCH
    if (const char* directory = getenv("MNN_PREFILL_BENCH_EXPORT")) {
        if (!impl.exportLayers(directory, path)) {
            return nullptr;
        }
    }
#endif
    MNN_PRINT("[CoreML] recognized %zu complete transformer layers, hidden=%d; double-buffered %s weights\n",
              impl.layers.size(), impl.layers[0].hidden, channelInt8 ? "per-channel INT8" : "FP16");
    return result;
}

std::shared_ptr<Tensor> CoreMLTransformer::run(const Tensor* input, const Tensor* cosine, const Tensor* sine,
                                               std::shared_ptr<KVCacheImport>& state) {
    auto& impl = *mImpl;
    std::lock_guard<std::mutex> lock(impl.mutex);
    const auto count = TensorUtils::getRawSize(input);
    const int h = hiddenSize();
    if (count % h || count / h < 1 || count / h > 2048 || input->getType() != halide_type_of<float>() ||
        cosine->getType() != halide_type_of<float>() || sine->getType() != halide_type_of<float>()) {
        return nullptr;
    }
    const int n = count / h;
    int t = 128;
    while (t < n) {
        t *= 2;
    }
    const int d = impl.layers[0].dim;
    if (TensorUtils::getRawSize(cosine) != n * d || TensorUtils::getRawSize(sine) != n * d) {
        MNN_ERROR("[CoreML] unsupported rotary input size\n");
        return nullptr;
    }
    auto x = halfTensor({t, h}), cos = halfTensor({t, d / 2}), sin = halfTensor({t, d / 2});
    if (!x || !cos || !sin) {
        return nullptr;
    }
    ::memset(x->host<void>(), 0, size_t(t) * h * 2);
    ::memset(cos->host<void>(), 0, size_t(t) * d);
    ::memset(sin->host<void>(), 0, size_t(t) * d);
    MNNQuantizeFP16(input->host<float>(), x->host<int16_t>(), count);
    for (int i = 0; i < n; ++i) {
        auto c = cosine->host<float>() + i * d;
        auto s = sine->host<float>() + i * d;
        if (!std::equal(c, c + d / 2, c + d / 2) || !std::equal(s, s + d / 2, s + d / 2)) {
            MNN_ERROR("[CoreML] rotary inputs must have repeated halves\n");
            return nullptr;
        }
        MNNQuantizeFP16(c, cos->host<int16_t>() + i * d / 2, d / 2);
        MNNQuantizeFP16(s, sin->host<int16_t>() + i * d / 2, d / 2);
    }
    std::vector<std::shared_ptr<CoreMLExecutorWrapper>> programs;
    bool fixed = false;
    double prepareFraction = 0.6;
#ifdef MNN_COREML_PREFILL_BENCH
    fixed = !impl.fixedDirectory.empty();
    if (impl.gpu) {
        prepareFraction = 0.4;
    }
#endif
    for (const auto& l : impl.layers) {
        std::ostringstream key;
        if (fixed) {
            key << programs.size() << ':';
        }
        key << t << ':' << l.hidden << ':' << l.heads << ':' << l.kv << ':' << l.dim << ':' << l.intermediate;
        for (auto eps : l.eps) {
            key << ':' << std::hexfloat << eps;
        }
        auto found = impl.programs.find(key.str());
        if (found == impl.programs.end()) {
            std::shared_ptr<CoreMLExecutorWrapper> program;
#ifdef MNN_COREML_PREFILL_BENCH
            if (fixed) {
                program = std::make_shared<CoreMLExecutorWrapper>(0);
                auto path = impl.fixedDirectory + "/" + std::to_string(t) + "-" + std::to_string(programs.size()) + ".mlmodelc";
                if (!program->loadBenchmarkModel(path)) {
                    return nullptr;
                }
            } else
#endif
            {
                program = compileLayer(l, t, impl.channelInt8);
            }
            if (!program) {
                return nullptr;
            }
            if (!fixed && impl.programs.size() >= 4) {
                impl.programs.erase(impl.programs.begin());
            }
            found = impl.programs.emplace(key.str(), std::move(program)).first;
            MNN_PRINT("[CoreML] compiled complete layer %s\n", key.str().c_str());
        }
        programs.push_back(found->second);
    }
    std::shared_ptr<KVCacheImport> pending(new KVCacheImport);
    pending->tokens = n;
    auto& slots = impl.slots;
    if (!fixed && !impl.prepare(0, slots[0])) {
        return nullptr;
    }
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    auto elapsed = [](Clock::time_point start) { return Milliseconds(Clock::now() - start).count(); };
    std::future<std::pair<bool, double>> prepare;
    double lastPredict = 0.0;
    double lastPrepare = 0.0;
    for (size_t i = 0; i < impl.layers.size(); ++i) {
        if (prepare.valid()) {
            const auto result = prepare.get();
            if (!result.first) {
                return nullptr;
            }
            lastPrepare = result.second;
        }
        if (!fixed && i + 1 < impl.layers.size()) {
            // Keep a margin for jitter and start immediately while learning the first two layers.
            double delay = 0.0;
            if (impl.channelInt8 && i > 1) {
                delay = std::max(0.0, std::min(lastPredict * prepareFraction, lastPredict - lastPrepare - 1.0));
            }
            const auto release = Clock::now() + Milliseconds(delay);
            prepare = std::async(std::launch::async, [&impl, &slots, i, release, delay, elapsed]() {
                if (delay > 0.0) {
                    std::this_thread::sleep_until(release);
                }
                const auto start = Clock::now();
                const bool ok = impl.prepare(i + 1, slots[(i + 1) % 2]);
                return std::make_pair(ok, elapsed(start));
            });
        }
        const auto& l = impl.layers[i];
        auto y = halfTensor({t, h});
        KVCacheImport::Layer kv;
        kv.key = halfTensor({t, l.kv, l.dim});
        kv.value = halfTensor({t, l.kv, l.dim});
        if (!y || !kv.key || !kv.value) {
            return nullptr;
        }
        std::vector<std::pair<const Tensor*, std::string>> inputs{
            {x.get(), "x"}, {cos.get(), "cosine"}, {sin.get(), "sine"}};
        CoreMLImageInputs images;
        const auto& slot = slots[i % 2];
        for (int j = 0; !fixed && j < 7; ++j) {
            if (impl.channelInt8) {
                images.emplace_back(weightName(j), slot.images[j].get());
                inputs.emplace_back(slot.scales[j].get(), std::string(weightName(j)) + "_scale");
            } else {
                inputs.emplace_back(slot.weights[j].get(), weightName(j));
            }
        }
        for (int j = 0; j < 4; ++j) {
            inputs.emplace_back(l.norms[j].get(), normName(j));
        }
        const auto predictStart = Clock::now();
        Interval interval(false, i);
        if (!programs[i]->invokModel(inputs, {{y.get(), "hidden"}, {kv.key.get(), "key"}, {kv.value.get(), "value"}},
                                     false, images)) {
            return nullptr;
        }
        lastPredict = elapsed(predictStart);
        if (n != t) {
            auto key = halfTensor({n, l.kv, l.dim}), value = halfTensor({n, l.kv, l.dim});
            if (!key || !value) {
                return nullptr;
            }
            ::memcpy(key->host<void>(), kv.key->host<void>(), size_t(n) * l.kv * l.dim * 2);
            ::memcpy(value->host<void>(), kv.value->host<void>(), size_t(n) * l.kv * l.dim * 2);
            kv.key = std::move(key);
            kv.value = std::move(value);
        }
        pending->layers.emplace(l.attention, std::move(kv));
        x = std::move(y);
    }
    if (n != t) {
        auto valid = halfTensor({n, h});
        if (!valid) {
            return nullptr;
        }
        ::memcpy(valid->host<void>(), x->host<void>(), size_t(n) * h * 2);
        x = std::move(valid);
    }
#ifdef MNN_COREML_PREFILL_BENCH
    MNN_PRINT("[CoreML bench] completed layers=%zu tokens=%d\n", impl.layers.size(), n);
#endif
    state = std::move(pending);
    return x;
}
} // namespace MNN
