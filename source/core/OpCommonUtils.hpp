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
#include "FileLoader.hpp"

namespace MNN {
struct Op;
struct CoreFunctions;

class MNN_PUBLIC OpCommonUtils {
#define USE_EXTERNAL_DATA(param) (param->external() && param->external()->size() > 1)
public:
    static Tensor::DimensionType convertDimType(MNN_DATA_FORMAT dimensionFormat);
    static bool supportDynamicInputMemory(MNNForwardType type);
    static void broastCastComputeDim(int* dims, int* stride, int* iStride0, int* iStride1, const Tensor* input0,
                                     const Tensor* input1, const Tensor* output);
    static std::vector<std::tuple<int, int, int>> computeReduceDims(const std::vector<Tensor*>& inputs, const Op* op);
    static void unravelIndexHelper(int32_t* coordinate, const int32_t* mod, int size,
                                   int indice);
    static int computeStride(int32_t* strides, const int* shape, int length);
    static void loadBlobData(FileLoader* loader, const Op* op, char* ptr, int size);

    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const Tensor* dest, int pack = 4, bool swapnc = false, bool swapcw = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const Tensor* dest, int pack = 4, bool swapnc = false);

    // Inside - Axis - Outside
    typedef std::tuple<int, int, int> SPLITS;
    static bool canBlitFast(const Tensor::InsideDescribe::Region& region, const SPLITS& srcSplits,
                            const SPLITS& dstSplits, int pack = 4, bool swapnc = false, bool swapcw = false);
    static void turnToPackRegion(const Tensor::InsideDescribe::Region& region, Tensor::InsideDescribe::Region& c4Region,
                                 const SPLITS& srcSplits, const SPLITS& dstSplits, int pack = 4, bool swapnc = false);
    static bool opNeedContent(const MNN::Op* op, int index);

    // For lowp CPU Backend
    static bool opCompabilityForLowp(const Op* op, int bytes);
    
    static void rasterInputReset(const std::vector<Tensor*>& inputs, Tensor* output);

    static void loadExternalDatas(FileLoader* loader, std::vector<char*> addrs, const int64_t* external);
    struct TensorConvertParameter {
        int batch;
        int channel;
        int area;
        int type;
    };

    // Detect if the region is a convert
    static void turnRegion2Convert(const Tensor::InsideDescribe::Region& region, const Tensor* dest, TensorConvertParameter& info);

    // Detect if the region is a transpose
    static bool isTranspose(const Tensor::InsideDescribe::Region& region, int& srcOne, int& dstOne);

    static bool computeMatMulSize(bool transposeA, bool transposeB, const Tensor* A, const Tensor* B, int& e, int& l, int& h);
    static Execution* createExecutionWithExternal(Backend* backend, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                                  const MNN::Op* op, FileLoader* externalFile, std::shared_ptr<BufferStorage>& tmpstore);
};
} // namespace MNN

#endif
