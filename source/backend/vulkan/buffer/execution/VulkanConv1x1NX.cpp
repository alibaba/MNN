#include "VulkanConv1x1NX.hpp"
#include "VulkanConv1x1General.hpp"
#include "VulkanNXMatMul.hpp"
#include "VulkanShaderMap.hpp"
#include "backend/vulkan/runtime/VulkanRuntime.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace MNN {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Two independent 64-bit hashes: a resident weight tensor is shared only when both match.
struct ContentHash {
    uint64_t h0 = 1469598103934665603ULL;
    uint64_t h1 = 0x9E3779B97F4A7C15ULL;

    // Word-at-a-time: the key only has to be unique inside this process, and mixing one byte per
    // iteration cost ~1.1 s of dependent multiplies over a 0.6B model's 440 MB of INT8 weights.
    void bytes(const void* data, size_t size) {
        const uint8_t* source = static_cast<const uint8_t*>(data);
        size_t i = 0;
        for (; i + sizeof(uint64_t) <= size; i += sizeof(uint64_t)) {
            uint64_t word;
            std::memcpy(&word, source + i, sizeof(word));
            h0 = (h0 ^ word) * 1099511628211ULL;
            h1 = (h1 ^ word) * 0x87C37B91114253D5ULL + 0xD6E8FEB86659FD93ULL;
        }
        for (; i < size; ++i) {
            h0 = (h0 ^ source[i]) * 1099511628211ULL;
            h1 = (h1 ^ source[i]) * 0x87C37B91114253D5ULL + 0xD6E8FEB86659FD93ULL;
        }
    }
    void value(uint64_t input) { bytes(&input, sizeof(input)); }
};

struct QuantPush {
    uint32_t M, rows, rowBase, K, padK, symmetric;
};
struct EpiloguePush {
    uint32_t M, rows, rowBase, N, K, padN, co4, activation, symmetric;
};
static_assert(sizeof(QuantPush) == 24 && sizeof(EpiloguePush) == 36, "NX convolution shader ABI");

// The INT8 graph costs the same GPU time for any M up to ~128 rows and grows ~0.32-0.52 us/row
// under sustained load above that, so a dispatch never covers more than kNxChunkRows rows and longer
// inputs are split into chunks. Every prompt length then shares one compiled graph shape and one
// prepared state per op, and the tail chunk pads its rows instead of needing a shape of its own.
// A 512-row bucket measures only 1.07x faster at pp512, which is not worth a second prepared shape
// even though sharing operands across layers cut its memory cost from +626 MB to about +15 MB.
constexpr uint32_t kNxChunkRows = 128;
// Below this row count the INT8 data-graph pipeline measures slower than VulkanConv1x1General.
constexpr uint64_t kNxMinRows = 2;
// Must match ROWS in nx_conv1x1_quant.comp, which packs this many rows per workgroup so that the
// [K/4, M, 4] activation reads stay coalesced.
constexpr uint32_t kNxQuantRowsPerGroup = 4;
// Must match local_size_x in nx_conv1x1_epilogue.comp. One ivec4 accumulator load per thread covers
// a whole output channel-quad, so a taller row block mainly buys four times fewer dispatch blocks
// and keeps a short tail chunk from wasting most of every block.
constexpr uint32_t kNxEpilogueRowsPerGroup = 32;
// Paired conv-compare runs (60 alternating NX/General pairs, M=2..128) show the win depends on ci,
// not on ci*co: every ci<=512 shape measured 0.37x..0.84x of General for co=512..4096, ci=576 was
// break-even at 1.00x, and ci>=640 measured 0.99x..1.96x. The graph dispatch itself degrades at
// small ci, so those shapes stay on VulkanConv1x1General.
constexpr int kNxMinChannels = 640;

bool nxDeviceSupported(VulkanBackend* backend) {
    if (!backend || !backend->useFP16()) {
        return false;
    }
    const auto& device = backend->getDevice();
    const auto& graph = device.getDataGraphInfo();
    const auto& limits = device.proty().limits;
    return graph.supported && graph.nxInt8Candidate && device.getInt8Support() && device.getFP16Support() &&
           limits.maxComputeWorkGroupInvocations >= 256 && limits.maxComputeWorkGroupSize[0] >= 32 &&
           limits.maxComputeWorkGroupSize[1] >= 64 && limits.maxComputeWorkGroupSize[2] >= 1 &&
           limits.maxComputeSharedMemorySize >= 4096 && limits.maxPushConstantsSize >= sizeof(EpiloguePush) &&
           limits.maxPerStageDescriptorStorageBuffers >= 5 && limits.maxDescriptorSetStorageBuffers >= 5;
}

bool quantSupported(const Convolution2DCommon* common, const ConvolutionCommon::Int8Common* quant, int ci, int co) {
    if (!common || !quant || ci <= 0 || ci > 131071 || co <= 0 || co > std::numeric_limits<int>::max() - 3 ||
        common->group() != 1 || common->kernelX() != 1 || common->kernelY() != 1 || common->strideX() != 1 ||
        common->strideY() != 1 || common->dilateX() != 1 || common->dilateY() != 1 || common->padX() != 0 ||
        common->padY() != 0 || common->hasOutputShape() || common->outputCount() != co ||
        (common->inputCount() != 0 && common->inputCount() != ci)) {
        return false;
    }
    if (common->padMode() != PadMode_CAFFE && common->padMode() != PadMode_SAME && common->padMode() != PadMode_VALID) {
        return false;
    }
    auto pads = common->pads();
    if (pads && pads->size() != 0 && pads->size() != 2 && pads->size() != 4) {
        return false;
    }
    if (pads) {
        for (uint32_t i = 0; i < pads->size(); ++i) {
            if (pads->Get(i) != 0) {
                return false;
            }
        }
    }
    auto outPads = common->outPads();
    if (outPads) {
        for (uint32_t i = 0; i < outPads->size(); ++i) {
            if (outPads->Get(i) != 0) {
                return false;
            }
        }
    }
    const uint64_t alphaCount = uint64_t(co) * (quant->asymmetric ? 2u : 1u);
    if (quant->originBits != 8 || quant->canUseInt4 || quant->canUseInt3 || quant->canUseInt2 ||
        (quant->quan && quant->quan->has_scaleInt()) || !quant->weight.get() ||
        uint64_t(quant->weight.size()) != uint64_t(ci) * uint64_t(co) || alphaCount != uint64_t(quant->alphaSize)) {
        return false;
    }
    // Corrected FP32 alpha takes precedence over the on-disk FP16 representation.
    return quant->alpha.get() ? uint64_t(quant->alpha.size()) >= alphaCount
                              : quant->alphaHalf.get() && uint64_t(quant->alphaHalf.size()) >= alphaCount;
}

bool compatibleTensors(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const Convolution2DCommon* common, int ci, int co) {
    if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0] || !outputs[0] || !common) {
        return false;
    }
    for (auto tensor : {inputs[0], outputs[0]}) {
        const auto type = tensor->getType();
        const auto des = TensorUtils::getDescribe(tensor);
        if (tensor->dimensions() != 4 || type.code != halide_type_float || type.lanes != 1 ||
            (type.bits != 16 && type.bits != 32) || des->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
            des->channel_pack_num != 4 || des->memoryType != Tensor::InsideDescribe::MEMORY_BACKEND ||
            des->applyQuant || !tensor->deviceId()) {
            return false;
        }
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input->batch() <= 0 || input->width() <= 0 || input->height() <= 0 || input->channel() != ci ||
        output->channel() != co || input->batch() != output->batch() || input->width() != output->width() ||
        input->height() != output->height()) {
        return false;
    }
    const auto pad = ConvolutionCommon::convolutionPadFull(input, output, common);
    return std::get<0>(pad) == 0 && std::get<1>(pad) == 0 && std::get<2>(pad) == 0 && std::get<3>(pad) == 0;
}

bool multiply(uint64_t a, uint64_t b, uint64_t& result) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

bool bufferFits(const VulkanBuffer* buffer, VkDeviceSize offset, VkDeviceSize bytes,
                const VkPhysicalDeviceLimits& limits) {
    const VkDeviceSize alignment = limits.minStorageBufferOffsetAlignment;
    return buffer && buffer->valid() && buffer->buffer() && bytes && bytes <= limits.maxStorageBufferRange &&
           offset <= buffer->size() && bytes <= buffer->size() - offset && (!alignment || offset % alignment == 0) &&
           offset % 8 == 0;
}

void memoryBarrier(VkCommandBuffer command, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(command, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

struct ComputeStage {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;

    VkResult prepare(VkDevice device, VulkanShaderMap& shaders, const char* name, uint32_t bindings, uint32_t pushBytes,
                     VkDescriptorPool pool, VkPipelineCache cache = VK_NULL_HANDLE) {
        const auto source = shaders.search(name);
        if (!source.first || source.second == 0 || source.second % sizeof(uint32_t) != 0) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        std::array<VkDescriptorSetLayoutBinding, 5> bindingInfo{};
        for (uint32_t i = 0; i < bindings; ++i) {
            bindingInfo[i].binding = i;
            bindingInfo[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindingInfo[i].descriptorCount = 1;
            bindingInfo[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo setInfo{};
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setInfo.bindingCount = bindings;
        setInfo.pBindings = bindingInfo.data();
        auto result = vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &setLayout);
        if (result != VK_SUCCESS) {
            return result;
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout);
        if (result != VK_SUCCESS) {
            return result;
        }
        // Own these small pipelines so a lost device is not hidden by the cached factory's pointer-only API.
        std::vector<uint32_t> code(source.second / sizeof(uint32_t));
        ::memcpy(code.data(), source.first, source.second);
        VkShaderModuleCreateInfo moduleInfo{};
        moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleInfo.codeSize = source.second;
        moduleInfo.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        result = vkCreateShaderModule(device, &moduleInfo, nullptr, &module);
        if (result != VK_SUCCESS) {
            return result;
        }
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = layout;
        result = vkCreateComputePipelines(device, cache, 1, &pipelineInfo, nullptr, &pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        if (result != VK_SUCCESS) {
            return result;
        }
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &setLayout;
        return vkAllocateDescriptorSets(device, &allocate, &set);
    }

    void destroy(VkDevice device) {
        if (pipeline)
            vkDestroyPipeline(device, pipeline, nullptr);
        if (layout)
            vkDestroyPipelineLayout(device, layout, nullptr);
        if (setLayout)
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    }

    void write(VkDevice device, uint32_t binding, VkBuffer buffer, VkDeviceSize bytes, VkDeviceSize offset = 0) {
        VkDescriptorBufferInfo info{buffer, offset, bytes};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void dispatch(VkCommandBuffer command, const void* push, uint32_t pushBytes, uint32_t x, uint32_t y) const {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
        vkCmdDispatch(command, x, y, 1);
    }
};

// Operand ranges for the data graph. The first convolution of a shape donates its own allocations and
// every later one binds its tensors into the same memory: a 0.6B model has 196 convolutions but only a
// handful of distinct padded shapes, which cuts the resident operand memory from ~176 MB to ~6 MB.
// Keyed by backend because two sessions sharing a runtime may execute concurrently.
struct NxScratch {
    explicit NxScratch(VulkanMemoryPool& pool) : mPool(pool) {}
    ~NxScratch() {
        if (activation.first)
            mPool.returnMemory(activation);
        if (accumulator.first)
            mPool.returnMemory(accumulator);
    }
    VulkanMemoryPool& mPool;
    MemChunk activation, accumulator;
};

std::mutex& nxScratchMutex() {
    static std::mutex guard;
    return guard;
}

std::unordered_map<std::string, std::weak_ptr<NxScratch>>& nxScratchEntries() {
    static std::unordered_map<std::string, std::weak_ptr<NxScratch>> entries;
    return entries;
}

std::shared_ptr<NxScratch> findNxScratch(const std::string& key) {
    std::lock_guard<std::mutex> guard(nxScratchMutex());
    auto& entries = nxScratchEntries();
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->second.expired())
            it = entries.erase(it);
        else
            ++it;
    }
    auto found = entries.find(key);
    return found == entries.end() ? nullptr : found->second.lock();
}

void publishNxScratch(const std::string& key, const std::shared_ptr<NxScratch>& scratch) {
    std::lock_guard<std::mutex> guard(nxScratchMutex());
    // emplace keeps the first donor; a racing second one just keeps its own ranges alive.
    nxScratchEntries().emplace(key, scratch);
}

} // namespace

struct VulkanConv1x1NX::Source {
    std::vector<uint8_t> commonData;
    std::vector<float> bias;
    int ci = 0;
    int co = 0;
    bool eligible = false;

    const Convolution2DCommon* common() const {
        return commonData.empty() ? nullptr : flatbuffers::GetRoot<Convolution2DCommon>(commonData.data());
    }
};

struct VulkanConv1x1NX::Weights {
    std::vector<float> metadata;
    // Content identity of the decoded weights, used to share one resident GPU tensor between the
    // independent execution instances (prefill/decode sessions) of the same convolution.
    uint64_t hash0 = 0;
    uint64_t hash1 = 0;
};

struct VulkanConv1x1NX::State {
    explicit State(VulkanBackend* backend) : device(backend->getDevice().get()), graph(new VulkanNXMatMul(backend)) {
        auto factory = backend->getPipelineFactory();
        if (factory && factory->pipelineCache()) {
            cache = factory->pipelineCache()->get();
            graph->setPipelineCache(cache);
        }
    }
    ~State() {
        if (pool)
            vkDestroyDescriptorPool(device, pool, nullptr);
        quant.destroy(device);
        epilogue.destroy(device);
    }
    VkDevice device;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::unique_ptr<VulkanNXMatMul> graph;
    std::shared_ptr<VulkanBuffer> params;
    std::shared_ptr<VulkanBuffer> sum;
    std::shared_ptr<VulkanBuffer> metadata;
    ComputeStage quant;
    ComputeStage epilogue;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    uint32_t rows = 0;
    uint32_t padK = 0;
    uint32_t padN = 0;
    // Keeps the shared operand ranges alive for as long as this graph is bound to them.
    std::shared_ptr<NxScratch> scratch;
    bool ready = false;
};

bool VulkanConv1x1NX::supported(VulkanBackend* backend, const Convolution2DCommon* common,
                                const ConvolutionCommon::Int8Common* quantCommon, int ci, int co) {
    if (!nxDeviceSupported(backend) || !quantSupported(common, quantCommon, ci, co)) {
        return false;
    }
    if (ci < kNxMinChannels) {
        return false;
    }
    const uint64_t padK = (uint64_t(ci) + 3u) & ~uint64_t(3u);
    const uint64_t padN = (uint64_t(co) + 3u) & ~uint64_t(3u);
    const uint64_t range = backend->getDevice().proty().limits.maxStorageBufferRange;
    return padK * padN <= range && padN * 4u * sizeof(float) <= range;
}

VulkanConv1x1NX::VulkanConv1x1NX(VulkanBackend* backend, const Convolution2DCommon* common, const float* bias, int ci,
                                 int co, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon)
    : VulkanBasicExecution(backend), mBackend(backend), mQuantCommon(std::move(quantCommon)) {
    auto source = std::make_shared<Source>();
    source->ci = ci;
    source->co = co;
    source->eligible = quantSupported(common, mQuantCommon.get(), ci, co);
    if (common) {
        Convolution2DCommonT copy;
        common->UnPackTo(&copy);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Convolution2DCommon::Pack(builder, &copy));
        source->commonData.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    }
    if (co > 0) {
        source->bias.resize(co, 0.0f);
        if (bias) {
            ::memcpy(source->bias.data(), bias, size_t(co) * sizeof(float));
        }
    }
    mSource = std::move(source);
}

VulkanConv1x1NX::VulkanConv1x1NX(VulkanBackend* backend, std::shared_ptr<const Source> source,
                                 std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon)
    : VulkanBasicExecution(backend),
      mBackend(backend),
      mSource(std::move(source)),
      mQuantCommon(std::move(quantCommon)) {}

VulkanConv1x1NX::~VulkanConv1x1NX() {
    // Recorded buffers may still reference descriptors, query pools and graph sessions.
    if (mState || mFallback) {
        mBackend->finish();
    }
    mState.reset();
    mFallback.reset();
}

bool VulkanConv1x1NX::onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) {
    (void)op;
    if (!backend || backend->type() != MNN_FORWARD_VULKAN || !mSource->eligible ||
        !nxDeviceSupported(static_cast<VulkanBackend*>(backend))) {
        return false;
    }
    if (!dst) {
        return true;
    }
    auto vkBackend = static_cast<VulkanBackend*>(backend);
    auto clone = new VulkanConv1x1NX(vkBackend, mSource, mQuantCommon);
    if (vkBackend->getDevice().get() == mBackend->getDevice().get()) {
        clone->mWeights = mWeights;
    }
    *dst = clone;
    return true;
}

bool VulkanConv1x1NX::prepareWeights(uint32_t padK, uint32_t padN) {
    if (mWeights) {
        return true;
    }
    const float* alpha = mQuantCommon->getAlphaFloat();
    if (!alpha) {
        mLastError = VK_ERROR_OUT_OF_HOST_MEMORY;
        return false;
    }
    auto weights = std::make_shared<Weights>();
    weights->metadata.resize(size_t(padN) * 4u, 0.0f);
    const int ci = mSource->ci;
    const int co = mSource->co;
    const auto raw = mQuantCommon->weight.get();
    // Hash the decoded source bytes: they determine both the transposed weights and the metadata,
    // so an equal key means an identical resident tensor can be reused without any upload.
    ContentHash hash;
    hash.value(uint64_t(padK));
    hash.value(uint64_t(padN));
    hash.value(uint64_t(ci));
    hash.value(uint64_t(co));
    hash.value(mQuantCommon->asymmetric ? 1ULL : 0ULL);
    for (int n = 0; n < co; ++n) {
        const int8_t* row = raw + size_t(n) * ci;
        // One pass over the weights serves both the per-channel sum and the identity hash.
        hash.bytes(row, size_t(ci));
        int32_t sum = 0;
        for (int k = 0; k < ci; ++k) {
            sum += row[k];
        }
        const float scale = alpha[mQuantCommon->asymmetric ? size_t(n) * 2u + 1u : size_t(n)];
        const float offset = mQuantCommon->asymmetric ? alpha[size_t(n) * 2u] : 0.0f;
        hash.bytes(&scale, sizeof(scale));
        hash.bytes(&offset, sizeof(offset));
        hash.bytes(mSource->bias.data() + n, sizeof(float));
        float* meta = weights->metadata.data() + size_t(n) * 4u;
        meta[0] = scale;
        meta[1] = offset;
        meta[2] = mSource->bias[n];
        // K <= 131071 keeps both the INT32 dot product and this FP32 integer sum exact.
        meta[3] = static_cast<float>(sum);
        if (!std::isfinite(meta[0]) || !std::isfinite(meta[1]) || !std::isfinite(meta[2])) {
            mLastError = VK_ERROR_FEATURE_NOT_PRESENT;
            return false;
        }
    }
    weights->hash0 = hash.h0;
    weights->hash1 = hash.h1;
    mWeights = std::move(weights);
    return true;
}

void VulkanConv1x1NX::transposeWeights(int8_t* output, uint32_t padK, uint32_t padN) const {
    const int ci = mSource->ci;
    const int co = mSource->co;
    const auto raw = mQuantCommon->weight.get();
    // Whole-buffer memset cost ~80 ms for a 0.6B model; the padding is usually empty because
    // channel counts are already multiples of four.
    for (uint32_t k = uint32_t(ci); k < padK; ++k) {
        std::memset(output + size_t(k) * padN, 0, padN);
    }
    for (uint32_t k = 0; k < uint32_t(ci) && co < int(padN); ++k) {
        std::memset(output + size_t(k) * padN + co, 0, padN - uint32_t(co));
    }
    // Blocked so that writes stay contiguous and each source cache line is reused across the block:
    // the naive per-element loop wrote with a padN stride and needed 2.0 s for 440 MB of weights.
    constexpr int kBlock = 64;
    for (int n0 = 0; n0 < co; n0 += kBlock) {
        const int nEnd = std::min(n0 + kBlock, co);
        for (int k0 = 0; k0 < ci; k0 += kBlock) {
            const int kEnd = std::min(k0 + kBlock, ci);
            for (int k = k0; k < kEnd; ++k) {
                int8_t* target = output + size_t(k) * padN + n0;
                for (int n = n0; n < nEnd; ++n) {
                    target[n - n0] = raw[size_t(n) * ci + k];
                }
            }
        }
    }
}

std::shared_ptr<VulkanNXWeightTensor> VulkanConv1x1NX::adoptSharedWeights(uint32_t padK, uint32_t padN) {
    if (!mWeights || mWeights->metadata.empty()) {
        return nullptr;
    }
    auto runtime = static_cast<const VulkanRuntime*>(mBackend->getRuntime());
    if (!runtime) {
        return nullptr;
    }
    char key[80];
    std::snprintf(key, sizeof(key), "nxw1-%016llx-%016llx-%u-%u", static_cast<unsigned long long>(mWeights->hash0),
                  static_cast<unsigned long long>(mWeights->hash1), padK, padN);
    if (auto found = runtime->findResident(key)) {
        return std::static_pointer_cast<VulkanNXWeightTensor>(found);
    }
    const auto transposeStart = Clock::now();
    const size_t bytes = size_t(padK) * padN;
    const auto& device = mBackend->getDevice();
    auto& pool = const_cast<VulkanMemoryPool&>(mBackend->getMemoryPool());
    auto backend = mBackend;
    auto source = this;
    double transposeMs = 0.0;
    bool created = false;
    auto value = runtime->acquireResident(key, [&]() -> std::shared_ptr<void> {
        created = true;
        auto resident = std::shared_ptr<VulkanNXWeightTensor>(new VulkanNXWeightTensor(device, pool));
        if (!resident->create(int(padK), int(padN))) {
            return nullptr;
        }
        // Transpose in cacheable host memory first: the tensor memory is host-visible but
        // write-combined, and blocked strided writes into it measured 2.2x slower than this
        // transpose plus one sequential copy.
        std::vector<int8_t> transposed(bytes);
        source->transposeWeights(transposed.data(), padK, padN);
        transposeMs = elapsedMs(transposeStart);
        if (!resident->upload(transposed.data(), bytes, backend)) {
            return nullptr;
        }
        return resident;
    });
    if (!value) {
        MNN_NX_LOG("NX_WEIGHT status=resident_unavailable padK=%u padN=%u fallback=per_shape_upload\n", padK, padN);
        return nullptr;
    }
    auto residentTensor = std::static_pointer_cast<VulkanNXWeightTensor>(value);
    MNN_NX_LOG("NX_WEIGHT status=%s padK=%u padN=%u bytes=%zu host_mapped=%d transpose_ms=%.3f create_upload_ms=%.3f\n",
               created ? "resident_created" : "resident_reused", padK, padN, bytes, int(residentTensor->mapped()),
               transposeMs, elapsedMs(transposeStart) - transposeMs);
    return std::static_pointer_cast<VulkanNXWeightTensor>(value);
}

VulkanConv1x1NX::State* VulkanConv1x1NX::findPreparedState(uint32_t bucket, uint32_t padK, uint32_t padN) {
    if (mState && mState->ready && mState->rows == bucket && mState->padK == padK && mState->padN == padN) {
        return mState.get();
    }
    return nullptr;
}

bool VulkanConv1x1NX::prepare(uint32_t rows, uint32_t bucket, uint32_t padK, uint32_t padN, double& weightMs,
                              double& computeSetupMs) {
    const auto weightStart = Clock::now();
    const bool weightsOk = prepareWeights(padK, padN);
    const double metadataMs = elapsedMs(weightStart);
    if (!weightsOk) {
        weightMs = elapsedMs(weightStart);
        return false;
    }
    auto resident = adoptSharedWeights(padK, padN);
    mResidentWeights = static_cast<bool>(resident);
    weightMs = elapsedMs(weightStart);
    MNN_NX_LOG("NX_WEIGHT_PHASE padK=%u padN=%u metadata_ms=%.3f total_ms=%.3f resident=%d\n", padK, padN, metadataMs,
               weightMs, int(mResidentWeights));
    // Release the previous shape before allocating the new one; the memory pool never shrinks.
    mState.reset(new State(mBackend));
    auto& state = *mState;
    state.rows = bucket;
    state.padK = padK;
    state.padN = padN;
    char scratchKey[96];
    std::snprintf(scratchKey, sizeof(scratchKey), "nxs-%p-%u-%u-%u", static_cast<const void*>(mBackend), bucket, padK,
                  padN);
    state.scratch = findNxScratch(scratchKey);
    if (state.scratch) {
        state.graph->setTensorAlias(0, state.scratch->activation);
        state.graph->setTensorAlias(2, state.scratch->accumulator);
        MNN_NX_LOG("NX_SCRATCH status=reused bucket=%u padK=%u padN=%u pool_mb=%.1f\n", bucket, padK, padN,
                   mBackend->getMemoryPool().computeSize());
    }
    if (!state.graph->prepareInt8(static_cast<int>(bucket), static_cast<int>(padK), static_cast<int>(padN),
                                  resident.get())) {
        mLastError = state.graph->lastError();
        if (mLastError == VK_SUCCESS)
            mLastError = VK_ERROR_FEATURE_NOT_PRESENT;
        return false;
    }
    if (!state.scratch) {
        state.scratch = std::make_shared<NxScratch>(const_cast<VulkanMemoryPool&>(mBackend->getMemoryPool()));
        state.scratch->activation = state.graph->takeMemory(0);
        state.scratch->accumulator = state.graph->takeMemory(2);
        if (state.scratch->activation.first && state.scratch->accumulator.first) {
            publishNxScratch(scratchKey, state.scratch);
            MNN_NX_LOG("NX_SCRATCH status=donated bucket=%u padK=%u padN=%u pool_mb=%.1f\n", bucket, padK, padN,
                       mBackend->getMemoryPool().computeSize());
        } else {
            MNN_NX_LOG("NX_SCRATCH status=private bucket=%u padK=%u padN=%u pool_mb=%.1f\n", bucket, padK, padN,
                       mBackend->getMemoryPool().computeSize());
        }
    }
    const VkDeviceSize logicalBytes[] = {VkDeviceSize(bucket) * padK, VkDeviceSize(padK) * padN,
                                         VkDeviceSize(bucket) * padN * sizeof(int32_t)};
    for (uint32_t i = 0; i < 3; ++i) {
        if (!state.graph->integerBuffer(i) || state.graph->integerBufferSize(i) < logicalBytes[i]) {
            mLastError = VK_ERROR_FEATURE_NOT_PRESENT;
            return false;
        }
    }
    auto start = Clock::now();
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto allocate = [&](VkDeviceSize bytes) {
        return std::make_shared<VulkanBuffer>(mBackend->getMemoryPool(), false, size_t(bytes), nullptr, usage,
                                              VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };
    state.params = allocate(VkDeviceSize(bucket) * 2u * sizeof(float));
    state.sum = allocate(VkDeviceSize(bucket) * sizeof(int32_t));
    state.metadata = allocate(VkDeviceSize(padN) * 4u * sizeof(float));
    if (!state.params->valid() || !state.sum->valid() || !state.metadata->valid()) {
        // VulkanBuffer exposes validity, not VkResult; check queue health before choosing a fallback.
        mLastError = vkQueueWaitIdle(mBackend->getDevice().acquireDefaultDevQueue());
        if (mLastError == VK_SUCCESS)
            mLastError = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        computeSetupMs = elapsedMs(start);
        return false;
    }
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    mLastError = vkCreateDescriptorPool(state.device, &poolInfo, nullptr, &state.pool);
    if (mLastError == VK_SUCCESS) {
        VulkanShaderMap shaders;
        mLastError = state.quant.prepare(state.device, shaders, "glsl_nx_conv1x1_quant_FP16_comp", 4, sizeof(QuantPush),
                                         state.pool, state.cache);
        if (mLastError == VK_SUCCESS) {
            mLastError = state.epilogue.prepare(state.device, shaders, "glsl_nx_conv1x1_epilogue_FP16_comp", 5,
                                                sizeof(EpiloguePush), state.pool, state.cache);
        }
    }
    computeSetupMs = elapsedMs(start);
    if (mLastError != VK_SUCCESS) {
        return false;
    }
    start = Clock::now();
    auto staging =
        mBackend->createHostBuffer(resident ? VkDeviceSize(state.metadata->size())
                                            : std::max(logicalBytes[1], VkDeviceSize(state.metadata->size())));
    auto upload = [&](const void* source, VkBuffer target, size_t bytes) {
        void* mapped = staging && staging->valid() ? staging->map() : nullptr;
        if (!mapped) {
            mLastError = vkQueueWaitIdle(mBackend->getDevice().acquireDefaultDevQueue());
            if (mLastError == VK_SUCCESS)
                mLastError = VK_ERROR_OUT_OF_HOST_MEMORY;
            return false;
        }
        std::memcpy(mapped, source, bytes);
        staging->unmap();
        mBackend->copyGPUToGPUBuffer(staging->buffer(), target, bytes, 0, 0);
        mLastError = vkQueueWaitIdle(mBackend->getDevice().acquireDefaultDevQueue());
        return mLastError == VK_SUCCESS;
    };
    // A resident tensor already holds this op's weights; only the per-state metadata still needs an upload.
    std::vector<int8_t> transposed;
    if (!resident) {
        transposed.resize(size_t(padK) * padN);
        transposeWeights(transposed.data(), padK, padN);
    }
    state.ready = (resident || upload(transposed.data(), state.graph->integerBuffer(1), logicalBytes[1])) &&
                  upload(mWeights->metadata.data(), state.metadata->buffer(), state.metadata->size());
    weightMs += elapsedMs(start);
    return state.ready;
}

ErrorCode VulkanConv1x1NX::fallback(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* command, const char* reason) {
    // A failed prepare leaves the state unusable; drop it so a later resize starts clean.
    mState.reset();
    if (mLastError == VK_ERROR_DEVICE_LOST) {
        MNN_ERROR("NX_CONV_ROUTE route=error reason=device_lost phase=%s vk_result=%d\n", reason, mLastError);
        return NO_EXECUTION;
    }
    if (mLastError == VK_ERROR_OUT_OF_HOST_MEMORY || mLastError == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        MNN_ERROR("NX_CONV_ROUTE route=error reason=out_of_memory phase=%s vk_result=%d\n", reason, mLastError);
        return OUT_OF_MEMORY;
    }
    return runGeneral(inputs, outputs, command, reason);
}

ErrorCode VulkanConv1x1NX::runGeneral(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                      const VulkanCommandPool::Buffer* command, const char* reason) {
    if (!mSource->eligible || !compatibleTensors(inputs, outputs, mSource->common(), mSource->ci, mSource->co) ||
        (mSource->ci & 3) != 0) {
        // General's per-channel implementation requires K % 4 == 0; it cannot rescue every NX shape.
        MNN_NX_LOG("NX_CONV_ROUTE route=skipped reason=%s general_compatible=0 vk_result=%d\n", reason, mLastError);
        return NOT_SUPPORT;
    }
    uint64_t spatial = 0, rows = 0;
    const auto& limits = mBackend->getDevice().proty().limits;
    const uint64_t padN = (uint64_t(mSource->co) + 31u) & ~uint64_t(31u);
    const uint64_t fpBytes = mBackend->useFP16() ? 2u : 4u;
    bool bounded =
        multiply(inputs[0]->width(), inputs[0]->height(), spatial) && multiply(spatial, inputs[0]->batch(), rows) &&
        rows <= uint64_t(std::numeric_limits<int>::max() - 63) && padN <= uint64_t(std::numeric_limits<int>::max() - 3);
    if (bounded) {
        for (auto tensor : {inputs[0], outputs[0]}) {
            const uint64_t channels = (uint64_t(tensor->channel()) + 3u) & ~uint64_t(3u);
            const uint64_t elements = rows * channels;
            const auto buffer = mBackend->getTensorBuffer(tensor);
            bounded = bounded && elements <= uint64_t(std::numeric_limits<int>::max()) / tensor->getType().bytes() &&
                      bufferFits(buffer.first, buffer.second, elements * fpBytes, limits);
        }
        const uint64_t padM = (rows + 63u) & ~uint64_t(63u);
        bounded = bounded && padN * mSource->ci <= limits.maxStorageBufferRange &&
                  padN * 2u * fpBytes <= limits.maxStorageBufferRange;
        if (rows == 1) {
            bounded = bounded && uint64_t(mSource->co) <= limits.maxComputeWorkGroupCount[0];
        } else {
            // General uses signed-size MNN temporaries, including four-byte allocation for its int16 tensors.
            const uint64_t maxElements =
                std::min<uint64_t>(std::numeric_limits<int>::max(), limits.maxStorageBufferRange) / 4u;
            bounded = bounded && padM * mSource->ci <= maxElements && padN * mSource->ci <= maxElements &&
                      padM / 64u <= limits.maxComputeWorkGroupCount[1] &&
                      padN / 32u <= limits.maxComputeWorkGroupCount[0] &&
                      uint64_t(mSource->ci) / 4u <= limits.maxComputeWorkGroupCount[0];
        }
    }
    if (!bounded) {
        MNN_NX_LOG("NX_CONV_ROUTE route=skipped reason=%s general_buffer_or_dispatch_bounds=0 vk_result=%d\n", reason,
                   mLastError);
        return NOT_SUPPORT;
    }
    if (!mFallback) {
        if (!mQuantCommon->getAlphaFloat()) {
            return OUT_OF_MEMORY;
        }
        mFallback.reset(new VulkanConv1x1General(mBackend, mSource->common(), mSource->bias.data(), mSource->ci,
                                                 mSource->co, mQuantCommon));
        const auto result = vkQueueWaitIdle(mBackend->getDevice().acquireDefaultDevQueue());
        if (result != VK_SUCCESS) {
            mLastError = result;
            MNN_ERROR("NX_CONV_ROUTE route=error reason=general_prepare vk_result=%d\n", result);
            return result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY ? OUT_OF_MEMORY
                                                                                                    : NO_EXECUTION;
        }
    }
    if (!mFallback->initialized()) {
        mFallback.reset();
        MNN_ERROR("NX_CONV_ROUTE route=error reason=general_init_failed\n");
        return NO_EXECUTION;
    }
    MNN_NX_LOG("NX_CONV_ROUTE route=General reason=%s vk_result=%d\n", reason, mLastError);
    return mFallback->onEncode(inputs, outputs, command);
}

ErrorCode VulkanConv1x1NX::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* command) {
    const auto resizeStart = Clock::now();
    if (!command || !command->get()) {
        return INVALID_VALUE;
    }
    if (mLastError == VK_ERROR_DEVICE_LOST) {
        return NO_EXECUTION;
    }
    // Resize owns descriptor refresh; the caller's recording buffer must never be submitted here.
    mBackend->finish();
    mLastError = vkQueueWaitIdle(mBackend->getDevice().acquireDefaultDevQueue());
    if (mLastError != VK_SUCCESS) {
        return fallback(inputs, outputs, command, "queue_wait");
    }
    if (!mSource->eligible || !nxDeviceSupported(mBackend)) {
        return fallback(inputs, outputs, command, "device_or_quantization");
    }
    if (!compatibleTensors(inputs, outputs, mSource->common(), mSource->ci, mSource->co)) {
        return fallback(inputs, outputs, command, "tensor_layout_or_padding");
    }
    const auto& device = mBackend->getDevice();
    const auto& limits = device.proty().limits;
    const uint32_t padK = (uint32_t(mSource->ci) + 3u) & ~3u;
    const uint32_t padN = (uint32_t(mSource->co) + 3u) & ~3u;
    uint64_t spatial = 0, rows = 0;
    if (!multiply(inputs[0]->width(), inputs[0]->height(), spatial) || !multiply(spatial, inputs[0]->batch(), rows) ||
        rows == 0 || rows > uint64_t(std::numeric_limits<int>::max()) ||
        kNxChunkRows > limits.maxComputeWorkGroupCount[0] ||
        (uint64_t(padN / 4u) + 7u) / 8u > limits.maxComputeWorkGroupCount[1] ||
        limits.maxComputeWorkGroupCount[2] == 0) {
        return fallback(inputs, outputs, command, "dispatch_bounds");
    }
    // Paired GPU measurements on Mali-G2-Ultra-NX for this op family: at M=1 the three-dispatch INT8
    // pipeline runs at 0.43-0.62x of General's fused kernel, while every M>=2 measured is 1.25-2.9x.
    // Single-row decode therefore stays on General; the NX state is kept for larger shapes.
    if (rows < kNxMinRows) {
        return runGeneral(inputs, outputs, command, "rows_below_nx_threshold");
    }
    const uint64_t bucket = kNxChunkRows;
    const uint32_t chunks = uint32_t((rows + bucket - 1) / bucket);
    // All shader addressing is uint32; also keep MNN's signed elementSize arithmetic representable.
    // Graph-owned buffers are sized by the bucket; the caller's tensors only ever hold `rows`.
    const uint64_t sizes[] = {bucket * padK,
                              uint64_t(padK) * padN,
                              bucket * padN * sizeof(int32_t),
                              bucket * 2u * sizeof(float),
                              bucket * sizeof(int32_t),
                              uint64_t(padN) * 4u * sizeof(float),
                              rows * padK * 2u,
                              rows * padN * 2u};
    uint64_t ownedBytes = 0;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        if (!sizes[i] || sizes[i] > limits.maxStorageBufferRange || sizes[i] > std::numeric_limits<size_t>::max()) {
            return fallback(inputs, outputs, command, "storage_bounds");
        }
        if (i < 6)
            ownedBytes += sizes[i];
    }
    if (sizes[6] / 2u > uint64_t(std::numeric_limits<int>::max()) / inputs[0]->getType().bytes() ||
        sizes[7] / 2u > uint64_t(std::numeric_limits<int>::max()) / outputs[0]->getType().bytes()) {
        return fallback(inputs, outputs, command, "tensor_index_overflow");
    }
    uint64_t largestHeap = 0;
    const auto& memory = device.memProty();
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            memory.memoryHeaps[i].size > largestHeap) {
            largestHeap = memory.memoryHeaps[i].size;
        }
    }
    if (ownedBytes > largestHeap) {
        return fallback(inputs, outputs, command, "device_memory_bounds");
    }
    const auto input = mBackend->getTensorBuffer(inputs[0]);
    const auto output = mBackend->getTensorBuffer(outputs[0]);
    if (!bufferFits(input.first, input.second, sizes[6], limits) ||
        !bufferFits(output.first, output.second, sizes[7], limits) || mBackend->getTensorSize(inputs[0]) < sizes[6] ||
        mBackend->getTensorSize(outputs[0]) < sizes[7]) {
        return fallback(inputs, outputs, command, "external_buffer_bounds");
    }
    auto* prepared = findPreparedState(uint32_t(bucket), padK, padN);
    const bool reused = prepared != nullptr;
    double weightMs = 0.0, computeSetupMs = 0.0, prepareMs = 0.0;
    auto logSetup = [&](bool success, double recordMs) {
        const auto& setup = mState->graph->setupTimes();
        uint64_t auxiliaryBytes = 0;
        for (auto buffer : {mState->params, mState->sum, mState->metadata}) {
            if (buffer)
                auxiliaryBytes += buffer->size();
        }
        const uint64_t cpuBytes = mWeights ? uint64_t(padK) * padN + mWeights->metadata.size() * sizeof(float) : 0;
        MNN_NX_LOG(
            "NX_CONV_SETUP status=%s M=%u M_bucket=%u chunks=%u K=%d N=%d padK=%u padN=%u reused=%d "
            "weight_upload_reused=%d "
            "weight_resident=%d "
            "graph_build_ms=%.3f tensor_buffer_ms=%.3f shader_module_ms=%.3f "
            "pipeline_layout_descriptor_ms=%.3f pipeline_compile_ms=%.3f session_init_ms=%.3f "
            "graph_prepare_total_ms=%.3f compute_setup_ms=%.3f weight_prepare_upload_ms=%.3f "
            "prepare_total_ms=%.3f resize_total_ms=%.3f record_ms=%.3f spirv_bytes=%zu "
            "tensor_bytes=%llu session_bytes=%llu aux_buffer_bytes=%llu cpu_weight_meta_bytes=%llu "
            "owned_buffer_session_bytes=%llu activation_bytes=%llu weight_bytes=%llu accumulator_bytes=%llu "
            "vk_result=%d\n",
            success ? "ready" : "failed", uint32_t(rows), uint32_t(bucket), chunks, mSource->ci, mSource->co, padK,
            padN, int(reused), int(reused), int(mResidentWeights), reused ? 0.0 : setup.graphBuildMs,
            reused ? 0.0 : setup.tensorBufferMs, reused ? 0.0 : setup.shaderModuleMs,
            reused ? 0.0 : setup.pipelineLayoutDescriptorMs, reused ? 0.0 : setup.pipelineCompileMs,
            reused ? 0.0 : setup.sessionInitMs, reused ? 0.0 : setup.totalMs, computeSetupMs, weightMs, prepareMs,
            elapsedMs(resizeStart), recordMs, setup.spirvBytes, static_cast<unsigned long long>(setup.tensorBytes),
            static_cast<unsigned long long>(setup.sessionBytes), static_cast<unsigned long long>(auxiliaryBytes),
            static_cast<unsigned long long>(cpuBytes),
            static_cast<unsigned long long>(setup.tensorBytes + setup.sessionBytes + auxiliaryBytes),
            static_cast<unsigned long long>(bucket * padK), static_cast<unsigned long long>(uint64_t(padK) * padN),
            static_cast<unsigned long long>(bucket * padN * sizeof(int32_t)), mLastError);
    };
    if (!reused) {
        // The row bucket is fixed, so only a K/N change forces a re-prepare.
        if (!prepare(uint32_t(rows), uint32_t(bucket), padK, padN, weightMs, computeSetupMs)) {
            prepareMs = elapsedMs(resizeStart);
            logSetup(false, 0.0);
            return fallback(inputs, outputs, command, "nx_prepare");
        }
    }
    mFallback.reset();
    auto& state = *mState;
    state.quant.write(state.device, 0, input.first->buffer(), sizes[6], input.second);
    state.quant.write(state.device, 1, state.graph->integerBuffer(0), sizes[0]);
    state.quant.write(state.device, 2, state.params->buffer(), sizes[3]);
    state.quant.write(state.device, 3, state.sum->buffer(), sizes[4]);
    state.epilogue.write(state.device, 0, state.graph->integerBuffer(2), sizes[2]);
    state.epilogue.write(state.device, 1, state.params->buffer(), sizes[3]);
    state.epilogue.write(state.device, 2, state.sum->buffer(), sizes[4]);
    state.epilogue.write(state.device, 3, state.metadata->buffer(), sizes[5]);
    state.epilogue.write(state.device, 4, output.first->buffer(), sizes[7], output.second);
    prepareMs = elapsedMs(resizeStart);
    const auto recordStart = Clock::now();

    // No fallible work remains past this point, so fallback cannot leave half a pipeline in the caller's buffer.
    const VkCommandBuffer cmd = command->get();
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    const auto common = mSource->common();
    const uint32_t activation = common->relu6() ? 2u : (common->relu() ? 1u : 0u);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const uint32_t rowBase = chunk * uint32_t(bucket);
        const uint32_t chunkRows = uint32_t(std::min<uint64_t>(rows - rowBase, bucket));
        // All chunks share one bucket-sized Params/SumAq, so this chunk's quant writes what the
        // previous chunk's epilogue is still reading. Both are compute, and the graph hand-off
        // barrier puts them on the same source stage, so it does not order them.
        if (chunk > 0) {
            memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        }
        const QuantPush quant{uint32_t(rows), chunkRows, rowBase, uint32_t(mSource->ci), padK, 0u};
#ifdef ENABLE_VULKAN_TIME_PROFILE
        // The three stages land on two different engines (quant/epilogue are compute shaders, the graph
        // is a data-graph dispatch), so the Execution-level scope cannot attribute them on its own.
        auto* profiler = mBackend->timeProfiler();
        std::unique_ptr<VulkanTimeProfileScope> stage;
        if (nullptr != profiler) {
            stage.reset(new VulkanTimeProfileScope(profiler, cmd, "nx_quant", VulkanTimeProfiler::Kind::Shader));
        }
#endif
        state.quant.dispatch(cmd, &quant, sizeof(quant),
                             (uint32_t(bucket) + kNxQuantRowsPerGroup - 1) / kNxQuantRowsPerGroup, 1);
#ifdef ENABLE_VULKAN_TIME_PROFILE
        stage.reset();
        if (nullptr != profiler) {
            stage.reset(new VulkanTimeProfileScope(profiler, cmd, "nx_graph", VulkanTimeProfiler::Kind::Shader));
        }
#endif
        // The graph's hand-off barriers cover quant->graph (AqPacked), graph->epilogue (Acc) and the
        // cross-chunk Acc hazard, since the previous epilogue is a prior command in their source stage.
        state.graph->encodeInt8(cmd);
#ifdef ENABLE_VULKAN_TIME_PROFILE
        stage.reset();
        if (nullptr != profiler) {
            stage.reset(new VulkanTimeProfileScope(profiler, cmd, "nx_epilogue", VulkanTimeProfiler::Kind::Shader));
        }
#endif
        const EpiloguePush epilogue{
            uint32_t(rows), chunkRows,  rowBase, uint32_t(mSource->co), uint32_t(mSource->ci), padN,
            padN / 4u,      activation, 0u};
        state.epilogue.dispatch(cmd, &epilogue, sizeof(epilogue),
                                (chunkRows + kNxEpilogueRowsPerGroup - 1) / kNxEpilogueRowsPerGroup,
                                (padN / 4u + 7u) / 8u);
#ifdef ENABLE_VULKAN_TIME_PROFILE
        stage.reset();
#endif
    }
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    logSetup(true, elapsedMs(recordStart));
    MNN_NX_LOG("NX_CONV_ROUTE route=NX M=%u M_bucket=%u chunks=%u K=%d N=%d reused=%d\n", uint32_t(rows),
               uint32_t(bucket), chunks, mSource->ci, mSource->co, int(reused));
    return NO_ERROR;
}

} // namespace MNN
