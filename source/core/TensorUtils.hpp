//
//  TensorUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include <MNN/Tensor.hpp>
#include "Backend.hpp"
#include "AutoStorage.h"
#include "Tensor_generated.h"
#define MNN_MAX_TENSOR_DIM 8

#ifdef CONSTANT
#undef CONSTANT
#endif // CONSTANT

namespace MNN {
struct TensorArrayAttr {
    // array size is dynamic or not
    bool isDynamicSize = false;
    // elemShape is identical or not
    bool isIdenticalShape = false;
    // the number of element
    uint32_t arraySize = 0;
    // the shape of element
    std::vector<std::vector<int>> elemShape;
};
struct QuantAttr {
    float scale;
    float zero = 0.0f;
    float min  = -127.0f;
    float max  = 127.0f;
};
struct Tensor::InsideDescribe {
    struct View {
        int32_t offset = 0;
        int32_t stride[3] = {1, 1, 1};
    };
    struct Region {
        View src;
        View dst;
        int32_t size[3] = {1, 1, 1};
        Tensor* origin;
    };
    struct pad {
        int32_t left = 0;
        int32_t right = 0;
        int32_t bottom = 0;
        int32_t top = 0;
    };
    enum MemoryType {
        /** The tensor's memory come from Backend */
        MEMORY_BACKEND = 0,

        /** host memory is owned by tensor or not */
        MEMORY_HOST,

        /** The tensor don't has memory */
        MEMORY_VIRTUAL,

        /** host memory is owned by tensor or not */
        MEMORY_OUTSIDE,
    };
    enum Usage {
        NORMAL,
        INPUT,
        OUTPUT,
        CONSTANT,
        /** Whether the tensor is a trainable parameter. Trainable parameter should be stored in a different area. */
        TRAINABLE,
    };
    // For Mask
    enum StageInfo {
        GEOMETRY_STAGE = 1,
        CONVERTED_STAGE = 1 << 1,
        COMPUTE_SHAPE_STAGE = 1 << 2,
        CONTENT_NOT_CHANGE = 1 << 3,
    };
    /** extra tensor info container */
    struct NativeInsideDescribe {
    public:
        /** dimension format */
        MNN_DATA_FORMAT dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        union {
            /** Serperate memory offset*/
            int offset;

            /** function used to free handle */
            void (*handleFreeFunction)(void*);
        } extra;
        MemoryType memoryType = MEMORY_BACKEND;
        std::weak_ptr<Command> rasterCommand;
        /** for DEVICE tensor only. */
        int useCount = 0;
        Usage usage = NORMAL;
        std::vector<Region> regions;
        halide_dimension_t dims[MNN_MAX_TENSOR_DIM];
        // TensorArray Attribute
        std::shared_ptr<TensorArrayAttr> tensorArrayAttr;
        // Tensor Quant Attribute
        std::shared_ptr<QuantAttr> quantAttr;
        // Only valid when quantAttr is not nullptr
        DataType type = DataType_DT_FLOAT;
        bool isMutable = true;
        int index = -1;
        int group = 0;
		int channel_pack_num = 4;
        bool support_pack16 = true;
        pad mPads;
        // For isMutable = false Tensor , determine whether the content can be convert to main backend
        uint32_t stageMask = 0;
    };
    std::shared_ptr<NativeInsideDescribe> mContent;
    SharedPtr<Backend::MemObj> mem;
    inline Backend* getBackend() const {
        return backend;
    }
    inline void setBackend(Backend* bn) {
        backend = bn;
    }
private:
    /** for DEVICE tensor only. backend used to manage tensor's device memory. */
    Backend* backend = nullptr;
};

typedef Tensor::InsideDescribe::Usage TensorUsage;

/** tensor utils */
class MNN_PUBLIC TensorUtils {
public:
    /**
     * @brief get extra tensor info.
     * @param tensor    given tensor.
     * @return extra tensor info.
     */
    static Tensor::InsideDescribe::NativeInsideDescribe* getDescribe(const Tensor* tensor);

    static Tensor::InsideDescribe* getDescribeOrigin(const Tensor* tensor);

    /**
     * @brief copy shape from source tensor to dest tensor.
     * @param source        shape prodiver tensor.
     * @param dest          shape consumer tensor.
     * @param copyFormat    copy data format or not.
     */
    static void copyShape(const Tensor* source, Tensor* dest, bool copyFormat = false, bool copyRef = false);

    /**
     * @brief set shape for dest tensor from a common int vector.
     * @param dest          shape consumer tensor.
     * @param alldims       dims info.
     */
    static void setShape(Tensor* dest, const std::vector<int>& alldims);

    /**
     * auto update tensor's strides according to extents and reorder flags.
     * @param tensor    given tensor.
     */
    static void setLinearLayout(Tensor* tensor);

    /**
     * @brief compare tensor to expected with tolerance.
     * @param compareTensor comparing tensor.
     * @param toTensor      expected tensor.
     * @param tolerance     tolerable error, any error less than this value will be ignored.
     *                      for integer types, compare with `abs(v1 - v2) > tolerance`;
     *                      for float types, see `overallTolerance`.
     * @param overall       for float types only. compare with `abs(v1 - v2) / max(abs(allExpectValues))` if true,
     *                      `abs(v1 - v2) / abs(v2)` otherwise.
     * @param printsError   print error data or not.
     * @param printsTensors print tensor data or not when meets error.
     * @return equals within tolerance or not.
     */
    static bool compareTensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance = 0,
                               bool overall = false, bool printsError = true, bool printsTensors = false);

    static void setupTensorInfo(const Tensor* tensor, Tensor* wrapTensor, MNN_DATA_FORMAT mMidFormat);
    static Tensor::InsideDescribe::Region makeFullSlice(Tensor* input);
    static bool regionIsFull(Tensor* input);
    static bool isCopyRegion(const Tensor::InsideDescribe::Region& region);
    static bool isTransposeRegion(const Tensor::InsideDescribe::Region& region);
    static bool isTileRegion(const Tensor::InsideDescribe::Region& region);
    static bool isDepthToSpaceRegions(const Tensor* output);
    static bool reshapeSlice(Tensor::InsideDescribe::Region& slice, int outside, int inside, int axis);
    
    class FuseRegionStatus;
    class MNN_PUBLIC FuseWrap {
    public:
        FuseWrap();
        ~ FuseWrap();
        bool match(const Tensor::InsideDescribe::Region& srcReg, const Tensor::InsideDescribe::Region& dstReg);
        void apply(const Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg);
    private:
        FuseRegionStatus* mStatus;
    };
    static void adjustTensorForCompability(Tensor* t);
    static Tensor::DimensionType getDimType(const Tensor* t);
    static std::vector<float> getQuantInfo(const Tensor* t);
    
    static size_t getRawSize(const Tensor* t);
    static void setRasterInputs(Command* cmd);
    
    static bool refTensorContent(Tensor* dst, const Tensor* src);

    static int getTensorChannelPack(const Tensor* tensor);

    static void setTensorChannelPack(const Tensor* tensor, int pack);

    static void setTensorSupportPack(const Tensor* tensor, bool flag);

    static void setTensorPad(const Tensor* tensor, int left, int right, int bottom, int top);
};
} // namespace MNN

#endif /* TensorDescribe_hpp */
