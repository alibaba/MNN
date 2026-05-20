//
//  Register.cpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "core/MusaBackend.hpp"
namespace MNN {
namespace MUSA {
class MusaRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        BackendConfig::MemoryMode memory       = BackendConfig::Memory_Normal;
        int device_id = 0;
        if (nullptr != info.user) {
            precision = info.user->precision;
            power     = info.user->power;
            memory    = info.user->memory;
            if (info.user->sharedContext != nullptr) {
                device_id = ((MNNDeviceContext *)info.user->sharedContext)->deviceId;
            }

        }
        auto backend = new MusaRuntimeWrapper(precision, power, memory, device_id);
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

bool placeholder = []() {
    static std::once_flag createOnce;
    std::call_once(createOnce, []() {
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_MUSA, new MusaRuntimeCreator, false);
    });
    return true;
}();

} // namespace MUSA
} // namespace MNN
