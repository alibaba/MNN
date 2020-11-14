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
class MNN_PUBLIC GeometryComputerUtils {
public:
    static void makeRaster(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static void addConvert(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer, GeometryComputer::Context& ctx);
    static Command makeCommand(const OpT* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    static Command makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output);
    static Command makeReduce(ReductionType type, Tensor* input0, Tensor* output);
    static Command makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output);
    static Command makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias = nullptr,
                              bool transposeA = false, bool transposeB = false);

    // offset, dstSize, originSize must be 3-int
    static void makeSliceRef(Tensor* dst, Tensor* src, const std::vector<int>& originSize,
                             const std::vector<int>& offset, const std::vector<int>& dstSize);

    static Tensor::InsideDescribe::Region makeRawAddressRef(Tensor* src, int srcOffset, int size, int dstOffset = 0);
    static void makeRawAddressRef(Tensor* dst, Tensor* src, int srcOffset, int size, int dstOffset = 0);
    static void buildConstantTensors(std::vector<Schedule::PipelineInfo>& infos, std::shared_ptr<Backend> backupBackend,
                                     bool netHold, std::vector<Tensor*>& constTensors,
                                     std::vector<Tensor*>& midConstTensors);
    static ErrorCode shapeComputeAndGeometryTransform(std::vector<Schedule::PipelineInfo>& infos, CommandBuffer& buffer,
                                                      GeometryComputer::Context& geoContext,
                                                      std::shared_ptr<Backend> backupBackend, bool geometry = true);
};
}; // namespace MNN

#endif
