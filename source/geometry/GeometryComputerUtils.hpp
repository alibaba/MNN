//
//  GeometryComputerUtils.hpp
//  MNN
//
//  Created by MNN on 2020/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GeometryComputerUtils_hpp
#define GeometryComputerUtils_hpp
#include "core/Schedule.hpp"
#include "geometry/GeometryComputer.hpp"
namespace MNN {
class FileLoader;
class GeometryComputerUtils {
public:
    MNN_PUBLIC static void makeRaster(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static void addConvert(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static std::shared_ptr<Command> makeCommand(flatbuffers::FlatBufferBuilder& builder, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    static std::shared_ptr<Command> makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output);
    static std::shared_ptr<Command> makeReduce(ReductionType type, Tensor* input0, Tensor* output);
    static std::shared_ptr<Command> makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output);
    static std::shared_ptr<Command> makeLayerNorm(Tensor* input0, Tensor* output, std::vector<int32_t> axis, float epsilon, std::vector<float> gamma, std::vector<float> beta, std::vector<int64_t> external, int group = 1, bool useRMS = false);
    static std::shared_ptr<Command> makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias = nullptr,
                              bool transposeA = false, bool transposeB = false);
    static flatbuffers::Offset<Op> makePool(flatbuffers::FlatBufferBuilder& builder, std::pair<int, int> kernel, std::pair<int, int> stride, PoolType type, MNN::PoolPadType pad, std::pair<int, int> pads, bool isglobal, AvgPoolCountType countType = AvgPoolCountType_DEFAULT);

    static Tensor::InsideDescribe::Region makeRawAddressRef(Tensor* src, int srcOffset, int size, int dstOffset = 0);
    static void makeRawAddressRef(Tensor* dst, Tensor* src, int srcOffset, int size, int dstOffset = 0);
    MNN_PUBLIC static int buildConstantTensors(std::vector<Schedule::OpCacheInfo>& infos);
    MNN_PUBLIC static ErrorCode shapeComputeAndGeometryTransform(FileLoader* external, std::vector<Schedule::OpCacheInfo>& infos,
                                                      GeometryComputer::Context& geoContext,
                                                      std::shared_ptr<Backend> backupBackend,
                                                                 Runtime::CompilerType compileType, bool skipShapeCompute = false, bool permitCodegen = false);
};
}; // namespace MNN

#endif
