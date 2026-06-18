#include "SafetensorModelRegistry.hpp"

#include <unordered_map>

#include <MNN/MNNDefine.h>

#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "writeFb.hpp"
#include "../common/CommonUtils.hpp"

namespace MNN {
namespace SafeTensors {

struct SafetensorModelRegistry::Impl {
    std::unordered_map<std::string, ModelBuilder> builders;
};

SafetensorModelRegistry* SafetensorModelRegistry::get() {
    static SafetensorModelRegistry gRegistry;
    if (!gRegistry.mImpl) {
        gRegistry.mImpl.reset(new Impl);
    }
    return &gRegistry;
}

void SafetensorModelRegistry::insert(const std::string& name, ModelBuilder builder) {
    if (name.empty() || builder == nullptr) {
        return;
    }
    if (!mImpl) {
        mImpl.reset(new Impl);
    }
    auto iter = mImpl->builders.find(name);
    if (iter != mImpl->builders.end()) {
        MNN_PRINT("SafetensorModelRegistry: override builder for %s\n", name.c_str());
    }
    mImpl->builders[name] = builder;
}

ModelBuilder SafetensorModelRegistry::find(const std::string& name) const {
    if (!mImpl) {
        return nullptr;
    }
    auto iter = mImpl->builders.find(name);
    if (iter == mImpl->builders.end()) {
        return nullptr;
    }
    return iter->second;
}

SafetensorModelRegister::SafetensorModelRegister(const char* name, ModelBuilder builder) {
    if (nullptr == name || builder == nullptr) {
        return;
    }
    SafetensorModelRegistry::get()->insert(name, builder);
}

MNN_PUBLIC void optimizeAndWrite(modelConfig& modelPath, std::unique_ptr<MNN::NetT>& netT) {
    if (nullptr == netT.get()) {
        return;
    }

    std::unique_ptr<MNN::OpT> metaOp(new MNN::OpT);
    metaOp->type = MNN::OpType_Extra;
    metaOp->main.value = new MNN::ExtraT;
    metaOp->main.type = MNN::OpParameter_Extra;
    metaOp->main.AsExtra()->type = "Meta";
    metaOp->main.AsExtra()->engine = "MNN";

    std::vector<std::string> expectedPass;
    CommonKit::loadCompress(modelPath);

    std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath, expectedPass);
    if (nullptr != newNet) {
        (void)writeFb(newNet, modelPath, std::move(metaOp));
    } else {
        MNN_ERROR("SafetensorModelRegistry: optimizeNet failed, skip writing %s\n", modelPath.MNNModel.c_str());
    }
}

} // namespace SafeTensors
} // namespace MNN
