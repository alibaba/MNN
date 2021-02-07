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
#include <set>
#include "MNN_generated.h"
#include "core/Command.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class GeometryComputer {
public:
    virtual ~GeometryComputer() {
        // Do nothing
    }
    class MNN_PUBLIC Context {
    public:
        Context(std::shared_ptr<Backend> allocBackend, bool permitVirtual = true);
        ~Context();

        void clear();
        void setBackend(Backend* backend);
        bool supportVirtual() const {
            return mPermitVirtual;
        }
        Tensor* getRasterCacheCreateRecurrse(Tensor* src, CommandBuffer& cmd);
        const std::vector<std::shared_ptr<Tensor>>& searchConst(const Op* op) const;
        std::shared_ptr<Tensor> allocConst(const Op* key, const std::vector<int>& shape, halide_type_t type,
                                           Tensor::DimensionType dimType = Tensor::TENSORFLOW);
        std::set<Tensor*> pOutputs;
    private:
        Tensor* getRasterCacheCreate(Tensor* src, CommandBuffer& cmd);
        std::shared_ptr<Tensor> getCachedTensor(Tensor* t);
        std::map<Tensor*, std::shared_ptr<Tensor>> mRasterCache;
        std::map<const Op*, std::vector<std::shared_ptr<Tensor>>> mConstTensors;
        std::vector<std::shared_ptr<Tensor>> mEmpty;
        bool mPermitVirtual;
        std::shared_ptr<Backend> mBackend;
        std::vector<uint8_t> mRasterOp;
    };
    static void init();
    MNN_PUBLIC static const GeometryComputer* search(int type);
    static Command makeRaster(Tensor* input, Tensor* output);
    static void registerGeometryComputer(std::shared_ptr<GeometryComputer> comp, std::vector<int> type);
    MNN_PUBLIC bool compute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            Context& context, CommandBuffer& cmd) const;

protected:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& cmd) const = 0;
};

class DefaultGeometryComputer : public GeometryComputer {
public:
    DefaultGeometryComputer() {
        // Do nothing
    }
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
