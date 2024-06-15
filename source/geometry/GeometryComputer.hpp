//
//  GeometryComputer.hpp
//  MNN
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GeometryComputer_hpp
#define GeometryComputer_hpp
#include <map>
#include <vector>
#include "MNN_generated.h"
#include "core/Command.hpp"
#include "core/TensorUtils.hpp"
#include "core/Backend.hpp"

namespace MNN {
class GeometryComputer {
public:
    virtual ~GeometryComputer() {
        // Do nothing
    }
    class MNN_PUBLIC Context {
    public:
        Context(int mask, std::shared_ptr<Backend> allocBackend, MNNForwardType type = MNN_FORWARD_CPU, BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal);
        ~Context();

        void clear();
        void setBackend(Backend* backend);
        void getRasterCacheCreateRecursive(Tensor* src, CommandBuffer& cmd);

        // If has cache, return. Otherwise create cache
        const std::vector<std::shared_ptr<Tensor>>& searchConst(const Op* op);
        std::shared_ptr<Tensor> allocConst(const Op* key, const std::vector<int>& shape, halide_type_t type,
                                           Tensor::DimensionType dimType = Tensor::TENSORFLOW);
        bool allocTensor(Tensor* tenosr);
        inline MNNForwardType forwardType() const {
            return mForwardType;
        }
        inline BackendConfig::PrecisionMode precisionType() const {
            return mPrecision;
        }
        inline bool support(int option) const {
            return mMask & option;
        }
        std::shared_ptr<BufferStorage> mRasterOp;
    private:
        void getRasterCacheCreate(Tensor* src, CommandBuffer& cmd);
        std::map<const Op*, std::vector<std::shared_ptr<Tensor>>> mConstTensors;
        std::vector<std::shared_ptr<Tensor>> mEmpty;
        std::vector<std::shared_ptr<Tensor>> mTempConstTensors;
        std::shared_ptr<Backend> mBackend;
        MNNForwardType mForwardType;
        BackendConfig::PrecisionMode mPrecision;
        TensorUtils::FuseWrap mFuseUtils;
        const int mMask;
    };
    static void init();
    MNN_PUBLIC static const GeometryComputer* search(int opType, Runtime::CompilerType compType);
    static void registerGeometryComputer(std::shared_ptr<GeometryComputer> comp, std::vector<int> type, Runtime::CompilerType compType = Runtime::Compiler_Geometry);

    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& cmd) const = 0;
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const {
        return false;
    }
    static bool ComputePermuteRegion(Tensor* input, Tensor* output, int* newshape, int shapeDim);
};

class DefaultGeometryComputer : public GeometryComputer {
public:
    DefaultGeometryComputer() {
        // Do nothing
    }
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             Context& context, CommandBuffer& cmd) const override;
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& cmd) const override;
};
void registerGeometryOps();

#define REGISTER_GEOMETRY(f, c)       \
    extern void ___##f##__##c##__() { \
        c();                          \
    }

} // namespace MNN

#endif
