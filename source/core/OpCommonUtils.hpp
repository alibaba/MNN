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
struct CoreFunctions;

class MNN_PUBLIC OpCommonUtils {
#define USE_EXTERNAL_DATA(param) (param->external() && param->external()->size() > 1)
public:
    static void broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0,
                                     const Tensor* input1, const Tensor* output);
    static std::vector<std::tuple<int, int, int>> computeReduceDims(const std::vector<Tensor*>& inputs, const Op* op);
    static void unravelIndexHelper(int32_t* coordinate, const int32_t* mod, int size,
                                   int indice);
    static int computeStride(int32_t* strides, const int* shape, int length);
    static void loadBlobData(Backend* backend, const Op* op, char* ptr, int size);

    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const Tensor* dest, int pack = 4, bool swapnc = false, bool swapcw = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const Tensor* dest, int pack = 4, bool swapnc = false);

    // Inside - Axis - Outside
    typedef std::tuple<int, int, int> SPLITS;
    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const SPLITS& srcSplits,
                            const SPLITS& dstSplits, int pack = 4, bool swapnc = false, bool swapcw = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const SPLITS& srcSplits, const SPLITS& dstSplits, int pack = 4, bool swapnc = false);
    static bool opNeedContent(int type, int index);

    // For lowp CPU Backend
    static bool opCompabilityForLowp(const Op* op);
    
    static void rasterInputReset(const std::vector<Tensor*>& inputs, Tensor* output);

    static void loadExternalData(Backend* backend, char* addr,  int64_t offset, int64_t size);
    static void loadExternalDatas(Backend* backend, std::vector<char*> addrs, const int64_t* external);
    static bool loadConvData(Backend* backend, const Op* op, std::unique_ptr<Tensor>& weight, std::unique_ptr<Tensor>& bias, int& weightSize, int& biasSize);
};
} // namespace MNN

#endif
