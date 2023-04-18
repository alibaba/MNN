//
//  Register.cpp
//  MNN
//
//  Created by MNN on 2020/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/CUDABackend.hpp"
namespace MNN {
namespace CUDA {
class CUDARuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        int device_id = 0;
        if (nullptr != info.user) {
            precision = info.user->precision;
            power     = info.user->power;

            if (info.user->sharedContext != nullptr) {
                device_id = ((MNNDeviceContext *)info.user->sharedContext)->deviceId;
            }

        }
        auto backend = new CUDARuntimeWrapper(precision, power, device_id);
        if (backend != nullptr) {
            if (!backend->isCreateError()) {
                return backend;
            } else {
                delete backend;
            }
        }
        return nullptr;
    }
};
static const auto __cuda_global_initializer = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CUDA, new CUDARuntimeCreator, false);
    return true;
}();

} // namespace CUDA
} // namespace MNN
