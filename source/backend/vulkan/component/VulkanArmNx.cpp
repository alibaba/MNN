#ifdef MNN_VULKAN_ARM_NX
#include "VulkanArmNx.hpp"
#include <spirv/unified1/spirv.hpp>
#include <spirv/unified1/TOSA.001000.1.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <initializer_list>
#include <map>

namespace MNN {
namespace {
std::atomic<uint64_t>& dispatchCount() {
    static std::atomic<uint64_t> count{0};
    return count;
}
// Emit the small public SPV_ARM_graph IR directly, without linking a model compiler.
class GraphModule {
public:
    uint32_t id() { return mNextId++; }
    void op(spv::Op opcode, std::initializer_list<uint32_t> operands) {
        mWords.push_back((static_cast<uint32_t>(operands.size() + 1) << 16) | opcode);
        mWords.insert(mWords.end(), operands);
    }
    void stringOp(spv::Op opcode, std::initializer_list<uint32_t> prefix, const char* string,
                  std::initializer_list<uint32_t> suffix = {}) {
        const size_t length = std::strlen(string) + 1;
        std::vector<uint32_t> words((length + 3) / 4, 0);
        std::memcpy(words.data(), string, length);
        mWords.push_back((static_cast<uint32_t>(1 + prefix.size() + words.size() + suffix.size()) << 16) | opcode);
        mWords.insert(mWords.end(), prefix);
        mWords.insert(mWords.end(), words.begin(), words.end());
        mWords.insert(mWords.end(), suffix);
    }
    std::vector<uint32_t> finish() {
        mWords[3] = mNextId;
        return std::move(mWords);
    }

private:
    uint32_t mNextId = 1;
    std::vector<uint32_t> mWords{spv::MagicNumber, 0x00010000, 0, 0, 0};
};

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) {
        return true;
    }
    MNN_ERROR("Vulkan ARM NX: %s failed (%d); using compute fallback\n", operation, result);
    return false;
}
} // namespace

std::vector<uint32_t> VulkanArmNx::matMulSpirv(uint32_t m, uint32_t k, uint32_t n, bool int8, uint32_t batch) {
    GraphModule module;
    module.op(spv::OpCapability, {spv::CapabilityShader});
    module.op(spv::OpCapability, {spv::CapabilityInt8});
    module.op(spv::OpCapability, {spv::CapabilityTensorsARM});
    module.op(spv::OpCapability, {spv::CapabilityGraphARM});
    module.stringOp(spv::OpExtension, {}, "SPV_ARM_tensors");
    module.stringOp(spv::OpExtension, {}, "SPV_ARM_graph");
    const auto tosa = module.id();
    module.stringOp(spv::OpExtInstImport, {tosa}, "TOSA.001000.1");
    module.op(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    uint32_t variables[3] = {module.id(), module.id(), module.id()};
    for (uint32_t binding = 0; binding < 3; ++binding) {
        module.op(spv::OpDecorate, {variables[binding], spv::DecorationDescriptorSet, 0});
        module.op(spv::OpDecorate, {variables[binding], spv::DecorationBinding, binding});
    }
    const auto u32 = module.id(), i8 = module.id(), valueType = module.id();
    module.op(spv::OpTypeInt, {u32, 32, 0});
    module.op(spv::OpTypeInt, {i8, 8, 1});
    if (int8) {
        module.op(spv::OpTypeInt, {valueType, 32, 1});
    } else {
        module.op(spv::OpTypeFloat, {valueType, 32});
    }
    std::map<uint32_t, uint32_t> constants;
    auto constant = [&](uint32_t value) {
        auto it = constants.find(value);
        if (it != constants.end())
            return it->second;
        auto result = module.id();
        module.op(spv::OpConstant, {u32, result, value});
        constants[value] = result;
        return result;
    };
    const auto zero = constant(0), one = constant(1), rank = constant(3);
    const auto shapeArray = module.id();
    module.op(spv::OpTypeArray, {shapeArray, u32, rank});
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> tensorTypes;
    auto tensorType = [&](uint32_t rows, uint32_t columns, uint32_t scalar) {
        const auto key = std::make_tuple(scalar, rows, columns);
        if (tensorTypes.count(key))
            return tensorTypes[key];
        const auto row = constant(rows), column = constant(columns), batches = constant(batch);
        const auto shape = module.id(), type = module.id();
        module.op(spv::OpConstantComposite, {shapeArray, shape, batches, row, column});
        module.op(spv::OpTypeTensorARM, {type, scalar, rank, shape});
        tensorTypes[key] = type;
        return type;
    };
    uint32_t types[3] = {tensorType(m, k, int8 ? i8 : valueType), tensorType(k, n, int8 ? i8 : valueType),
                         tensorType(m, n, valueType)};
    const auto zpArray = module.id(), zpShape = module.id(), zpType = module.id(), zp = module.id();
    module.op(spv::OpTypeArray, {zpArray, u32, one});
    module.op(spv::OpConstantComposite, {zpArray, zpShape, one});
    module.op(spv::OpTypeTensorARM, {zpType, i8, one, zpShape});
    module.op(spv::OpConstantNull, {zpType, zp});
    std::map<uint32_t, uint32_t> pointers;
    for (int i = 0; i < 3; ++i) {
        auto& pointer = pointers[types[i]];
        if (!pointer) {
            pointer = module.id();
            module.op(spv::OpTypePointer, {pointer, spv::StorageClassUniformConstant, types[i]});
        }
        module.op(spv::OpVariable, {pointer, variables[i], spv::StorageClassUniformConstant});
    }
    const auto graphType = module.id(), graph = module.id();
    module.op(spv::OpTypeGraphARM, {graphType, 2, types[0], types[1], types[2]});
    module.stringOp(spv::OpGraphEntryPointARM, {graph}, "main", {variables[0], variables[1], variables[2]});
    module.op(spv::OpGraphARM, {graphType, graph});
    const auto a = module.id(), b = module.id(), result = module.id();
    module.op(spv::OpGraphInputARM, {types[0], a, zero});
    module.op(spv::OpGraphInputARM, {types[1], b, one});
    module.op(spv::OpExtInst, {types[2], result, tosa, TOSAMATMUL, a, b, zp, zp});
    module.op(spv::OpGraphSetOutputARM, {result, zero});
    module.op(spv::OpGraphEndARM, {});
    return module.finish();
}

std::unique_ptr<VulkanArmNx> VulkanArmNx::createMatMul(const VulkanDevice& device, uint32_t m, uint32_t k, uint32_t n,
                                                       bool int8, uint32_t batch) {
    if (!device.supportsArmNx() || !device.getInt8Support() || !m || !k || !n || !batch ||
        uint64_t(batch) > UINT32_MAX / 4 / uint64_t(m) / k || uint64_t(batch) > UINT32_MAX / 4 / uint64_t(k) / n ||
        uint64_t(batch) > UINT32_MAX / 4 / uint64_t(m) / n) {
        return nullptr;
    }
    std::unique_ptr<VulkanArmNx> graph(new VulkanArmNx(device));
    if (!graph->init(m, k, n, int8, batch))
        return nullptr;
    return graph;
}

bool VulkanArmNx::allocate(const VkMemoryRequirements& requirements, VkDeviceMemory& memory) {
    const auto& properties = mDevice.memProty();
    uint32_t type = 0;
    for (; type < properties.memoryTypeCount; ++type) {
        if (requirements.memoryTypeBits & (1u << type))
            break;
    }
    if (type == properties.memoryTypeCount)
        return false;
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = requirements.size;
    info.memoryTypeIndex = type;
    return check(vkAllocateMemory(mDevice.get(), &info, nullptr, &memory), "allocate graph memory");
}

bool VulkanArmNx::createTensor(size_t index, uint32_t batch, int64_t rows, int64_t columns, VkFormat format) {
    auto& resource = mResources[index];
    resource.dimensions = {{batch, rows, columns}};
    resource.description = {VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM};
    auto& description = resource.description;
    description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
    description.format = format;
    description.dimensionCount = 3;
    description.pDimensions = resource.dimensions.data();
    description.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    resource.bytes = (static_cast<VkDeviceSize>(batch) * rows * columns * (format == VK_FORMAT_R8_SINT ? 1 : 4) + 3) &
                     ~VkDeviceSize(3);
    VkTensorCreateInfoARM info{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
    info.pDescription = &description;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(mCreateTensor(mDevice.get(), &info, nullptr, &resource.tensor), "create tensor"))
        return false;
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = resource.bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(mDevice.get(), &buffer, nullptr, &resource.buffer), "create alias buffer"))
        return false;
    VkTensorMemoryRequirementsInfoARM tensorInfo{VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
    tensorInfo.tensor = resource.tensor;
    VkMemoryRequirements2 tensorRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    mGetTensorMemoryRequirements(mDevice.get(), &tensorInfo, &tensorRequirements);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(mDevice.get(), resource.buffer, &requirements);
    requirements.size = std::max(requirements.size, tensorRequirements.memoryRequirements.size);
    requirements.memoryTypeBits &= tensorRequirements.memoryRequirements.memoryTypeBits;
    if (!allocate(requirements, resource.memory))
        return false;
    VkBindTensorMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
    bind.tensor = resource.tensor;
    bind.memory = resource.memory;
    if (!check(mBindTensorMemory(mDevice.get(), 1, &bind), "bind tensor memory") ||
        !check(vkBindBufferMemory(mDevice.get(), resource.buffer, resource.memory, 0), "bind alias memory"))
        return false;
    VkTensorViewCreateInfoARM view{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};
    view.tensor = resource.tensor;
    view.format = format;
    return check(mCreateTensorView(mDevice.get(), &view, nullptr, &resource.view), "create tensor view");
}

bool VulkanArmNx::init(uint32_t m, uint32_t k, uint32_t n, bool int8, uint32_t batch) {
#define LOAD_GRAPH(member, name)                                                          \
    do {                                                                                  \
        member = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(mDevice.get(), #name)); \
        if (!member) {                                                                    \
            MNN_ERROR("Vulkan ARM NX: missing %s\n", #name);                      \
            return false;                                                                 \
        }                                                                                 \
    } while (false)
    LOAD_GRAPH(mCreateTensor, vkCreateTensorARM);
    LOAD_GRAPH(mDestroyTensor, vkDestroyTensorARM);
    LOAD_GRAPH(mGetTensorMemoryRequirements, vkGetTensorMemoryRequirementsARM);
    LOAD_GRAPH(mBindTensorMemory, vkBindTensorMemoryARM);
    LOAD_GRAPH(mCreateTensorView, vkCreateTensorViewARM);
    LOAD_GRAPH(mDestroyTensorView, vkDestroyTensorViewARM);
    LOAD_GRAPH(mCreatePipelines, vkCreateDataGraphPipelinesARM);
    LOAD_GRAPH(mCreateSession, vkCreateDataGraphPipelineSessionARM);
    LOAD_GRAPH(mDestroySession, vkDestroyDataGraphPipelineSessionARM);
    LOAD_GRAPH(mGetSessionBindPoints, vkGetDataGraphPipelineSessionBindPointRequirementsARM);
    LOAD_GRAPH(mGetSessionMemoryRequirements, vkGetDataGraphPipelineSessionMemoryRequirementsARM);
    LOAD_GRAPH(mBindSessionMemory, vkBindDataGraphPipelineSessionMemoryARM);
    LOAD_GRAPH(mDispatch, vkCmdDispatchDataGraphARM);
    LOAD_GRAPH(mBarrier, vkCmdPipelineBarrier2);
#undef LOAD_GRAPH
    if (!createTensor(0, batch, m, k, int8 ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SFLOAT) ||
        !createTensor(1, batch, k, n, int8 ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SFLOAT) ||
        !createTensor(2, batch, m, n, int8 ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_SFLOAT))
        return false;
    VkDescriptorSetLayoutBinding bindings[3]{};
    VkDataGraphPipelineResourceInfoARM resources[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        resources[i] = {VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM};
        resources[i].binding = i;
        resources[i].pNext = &mResources[i].description;
    }
    VkDescriptorSetLayoutCreateInfo setLayout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayout.bindingCount = 3;
    setLayout.pBindings = bindings;
    if (!check(vkCreateDescriptorSetLayout(mDevice.get(), &setLayout, nullptr, &mSetLayout),
               "create descriptor layout"))
        return false;
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &mSetLayout;
    if (!check(vkCreatePipelineLayout(mDevice.get(), &layout, nullptr, &mLayout), "create pipeline layout"))
        return false;
    auto code = matMulSpirv(m, k, n, int8, batch);
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = code.size() * sizeof(uint32_t);
    shader.pCode = code.data();
    if (!check(vkCreateShaderModule(mDevice.get(), &shader, nullptr, &mShader), "create graph module"))
        return false;
    VkDataGraphPipelineShaderModuleCreateInfoARM module{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM};
    module.module = mShader;
    module.pName = "main";
    VkDataGraphPipelineCreateInfoARM pipeline{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM};
    pipeline.pNext = &module;
    pipeline.layout = mLayout;
    pipeline.resourceInfoCount = 3;
    pipeline.pResourceInfos = resources;
    if (!check(mCreatePipelines(mDevice.get(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline, nullptr, &mPipeline),
               "compile MATMUL graph"))
        return false;
    VkDataGraphPipelineSessionCreateInfoARM session{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM};
    session.dataGraphPipeline = mPipeline;
    if (!check(mCreateSession(mDevice.get(), &session, nullptr, &mSession), "create graph session"))
        return false;
    VkDataGraphPipelineSessionBindPointRequirementsInfoARM query{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM};
    query.session = mSession;
    uint32_t count = 0;
    if (!check(mGetSessionBindPoints(mDevice.get(), &query, &count, nullptr), "query session bind points"))
        return false;
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> points(count);
    for (auto& point : points)
        point.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
    if (!check(mGetSessionBindPoints(mDevice.get(), &query, &count, points.data()), "query session bind points"))
        return false;
    for (const auto& point : points) {
        if (point.bindPointType != VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM)
            return false;
        for (uint32_t object = 0; object < point.numObjects; ++object) {
            VkDataGraphPipelineSessionMemoryRequirementsInfoARM memoryInfo{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM};
            memoryInfo.session = mSession;
            memoryInfo.bindPoint = point.bindPoint;
            memoryInfo.objectIndex = object;
            VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
            mGetSessionMemoryRequirements(mDevice.get(), &memoryInfo, &requirements);
            if (!requirements.memoryRequirements.size)
                continue;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (!allocate(requirements.memoryRequirements, memory))
                return false;
            mSessionMemory.push_back(memory);
            VkBindDataGraphPipelineSessionMemoryInfoARM bind{
                VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM};
            bind.session = mSession;
            bind.bindPoint = point.bindPoint;
            bind.objectIndex = object;
            bind.memory = memory;
            if (!check(mBindSessionMemory(mDevice.get(), 1, &bind), "bind session memory"))
                return false;
        }
    }
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_TENSOR_ARM, 3};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    if (!check(vkCreateDescriptorPool(mDevice.get(), &pool, nullptr, &mDescriptorPool), "create descriptor pool"))
        return false;
    VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set.descriptorPool = mDescriptorPool;
    set.descriptorSetCount = 1;
    set.pSetLayouts = &mSetLayout;
    if (!check(vkAllocateDescriptorSets(mDevice.get(), &set, &mSet), "allocate descriptor set"))
        return false;
    for (uint32_t i = 0; i < 3; ++i) {
        VkWriteDescriptorSetTensorARM tensor{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM};
        tensor.tensorViewCount = 1;
        tensor.pTensorViews = &mResources[i].view;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.pNext = &tensor;
        write.dstSet = mSet;
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        vkUpdateDescriptorSets(mDevice.get(), 1, &write, 0, nullptr);
    }
    MNN_PRINT("Vulkan ARM NX: MATMUL batch=%u %ux%ux%u %s\n", batch, m, k, n, int8 ? "INT8->INT32" : "FP32");
    return true;
}

void VulkanArmNx::barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                              VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = srcStage;
    memory.srcAccessMask = srcAccess;
    memory.dstStageMask = dstStage;
    memory.dstAccessMask = dstAccess;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    mBarrier(cmd, &dependency);
}

void VulkanArmNx::encode(VkCommandBuffer cmd, VkBuffer a, VkDeviceSize aOffset, VkBuffer b, VkDeviceSize bOffset,
                             VkBuffer output, VkDeviceSize outputOffset) const {
    dispatchCount().fetch_add(1, std::memory_order_relaxed);
    // Global barriers also cover tensor/buffer aliases and session reuse on the same queue.
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferCopy aCopy{aOffset, 0, mResources[0].bytes}, bCopy{bOffset, 0, mResources[1].bytes};
    vkCmdCopyBuffer(cmd, a, mResources[0].buffer, 1, &aCopy);
    vkCmdCopyBuffer(cmd, b, mResources[1].buffer, 1, &bCopy);
    barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
            VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, mPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, mLayout, 0, 1, &mSet, 0, nullptr);
    mDispatch(cmd, mSession, nullptr);
    barrier(cmd, VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
            VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferCopy outputCopy{0, outputOffset, mResources[2].bytes};
    vkCmdCopyBuffer(cmd, mResources[2].buffer, output, 1, &outputCopy);
    barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

uint64_t VulkanArmNx::recordedDispatchCount() {
    return dispatchCount().load(std::memory_order_relaxed);
}

VulkanArmNx::~VulkanArmNx() {
    const auto device = mDevice.get();
    if (mDescriptorPool)
        vkDestroyDescriptorPool(device, mDescriptorPool, nullptr);
    if (mSession)
        mDestroySession(device, mSession, nullptr);
    for (auto memory : mSessionMemory)
        vkFreeMemory(device, memory, nullptr);
    if (mPipeline)
        vkDestroyPipeline(device, mPipeline, nullptr);
    if (mShader)
        vkDestroyShaderModule(device, mShader, nullptr);
    if (mLayout)
        vkDestroyPipelineLayout(device, mLayout, nullptr);
    if (mSetLayout)
        vkDestroyDescriptorSetLayout(device, mSetLayout, nullptr);
    for (const auto& resource : mResources) {
        if (resource.view)
            mDestroyTensorView(device, resource.view, nullptr);
        if (resource.tensor)
            mDestroyTensor(device, resource.tensor, nullptr);
        if (resource.buffer)
            vkDestroyBuffer(device, resource.buffer, nullptr);
        if (resource.memory)
            vkFreeMemory(device, resource.memory, nullptr);
    }
}
} // namespace MNN
#endif
