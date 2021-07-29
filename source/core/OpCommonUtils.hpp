//
//  OpCommonUtils.hpp
//  MNN
//
//  Created by MNN on 2020/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpCommonUtils_hpp
#define OpCommonUtils_hpp
#include <MNN/Tensor.hpp>
#include "TensorUtils.hpp"
namespace MNN {
struct Op;
class MNN_PUBLIC OpCommonUtils {
public:
    static void broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0,
                                     const Tensor* input1, const Tensor* output);
    static std::vector<std::tuple<int, int, int>> computeReduceDims(const std::vector<Tensor*>& inputs, const Op* op);
    static void unravelIndexHelper(std::vector<int32_t>& coordinate, const std::vector<int32_t>& mod, int size,
                                   int indice);
    static int computeStride(int32_t* strides, const int* shape, int length);
    static void* blobData(const Op* op);

    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const Tensor* dest, int pack = 4, bool swapnc = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const Tensor* dest, int pack = 4, bool swapnc = false);

    // Inside - Axis - Outside
    typedef std::tuple<int, int, int> SPLITS;
    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const SPLITS& srcSplits,
                            const SPLITS& dstSplits, int pack = 4, bool swapnc = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const SPLITS& srcSplits, const SPLITS& dstSplits, int pack = 4, bool swapnc = false);
    static bool opNeedContent(int type, int index);

    // For lowp CPU Backend
    static bool opCompabilityForLowp(const Op* op);
    static std::pair<bool,DataType> getQuantInfo(const std::vector<Tensor*>& inputs);

    // sparse common functions
    static void statisticWeightSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, const float* data, size_t h, size_t l,  int sparseBlockOC);
    static void fillRandValueAsSparsity(size_t& weightNNZElement, size_t& weightBlockNumber, float* data, int oc, int reduceDimLength, float sparsity, int sparseBlockOC);

    static bool checkAllZeros(const float* source, size_t rowDimLength, int blockRow, int blockCol);


};
} // namespace MNN

#endif
