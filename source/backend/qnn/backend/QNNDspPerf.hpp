//
// QNNDspPerf.hpp
// MNN
//

#ifndef QNN_DSP_PERF_HPP
#define QNN_DSP_PERF_HPP

#include "QnnInterface.h"
#include <memory>

namespace MNN {
namespace QNN {

class QNNDspPerf {
public:
    static std::unique_ptr<QNNDspPerf> create(
        const QNN_INTERFACE_VER_TYPE* qnnInterface);
    ~QNNDspPerf();

    bool setPowerConfigBurst();
    bool setPowerConfigBalanced();

private:
    explicit QNNDspPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface);
    bool initialize();

private:
    const QNN_INTERFACE_VER_TYPE* mQnnInterface = nullptr;
    void* mInfrastructure = nullptr;
    uint32_t mPowerConfigId = 0;
    bool mInitialized = false;
};

} // end namespace QNN
} // end namespace MNN

#endif // QNN_DSP_PERF_HPP
