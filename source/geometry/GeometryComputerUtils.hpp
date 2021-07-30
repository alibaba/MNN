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
class GeometryComputerUtils {
public:
    MNN_PUBLIC static void makeRaster(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static void addConvert(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static Command makeCommand(flatbuffers::FlatBufferBuilder& builder, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    static Command makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output);
    static Command makeReduce(ReductionType type, Tensor* input0, Tensor* output);
    static Command makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output);
    static Command makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias = nullptr,
                              bool transposeA = false, bool transposeB = false);
    static flatbuffers::Offset<Op> makePool(flatbuffers::FlatBufferBuilder& builder, std::pair<int, int> kernel, std::pair<int, int> stride, PoolType type, MNN::PoolPadType pad, std::pair<int, int> pads, bool isglobal, AvgPoolCountType countType = AvgPoolCountType_DEFAULT);

    // offset, dstSize, originSize must be 3-int
    static void makeSliceRef(Tensor* dst, Tensor* src, const std::vector<int>& originSize,
                             const std::vector<int>& offset, const std::vector<int>& dstSize);

    static Tensor::InsideDescribe::Region makeRawAddressRef(Tensor* src, int srcOffset, int size, int dstOffset = 0);
    static void makeRawAddressRef(Tensor* dst, Tensor* src, int srcOffset, int size, int dstOffset = 0);
    MNN_PUBLIC static void buildConstantTensors(std::vector<Schedule::PipelineInfo>& infos, std::shared_ptr<Backend> backupBackend,
                                     bool netHold,
                                     std::vector<Tensor*>& midConstTensors);
    MNN_PUBLIC static ErrorCode shapeComputeAndGeometryTransform(std::vector<Schedule::PipelineInfo>& infos, CommandBuffer& buffer,
                                                      GeometryComputer::Context& geoContext,
                                                      std::shared_ptr<Backend> backupBackend,
                                                                 Runtime::CompilerType compileType);
};
}; // namespace MNN

#endif
