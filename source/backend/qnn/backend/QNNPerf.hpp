//
//  QNNPerf.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef QNN_PERF_HPP
#define QNN_PERF_HPP

#include "QNNUtils.hpp"
#include "HTP/QnnHtpDevice.h"

namespace MNN {
namespace QNN {

class QNNPerf {
public:
    static std::unique_ptr<QNNPerf> create(const QNN_INTERFACE_VER_TYPE * qnnInterface) {return std::unique_ptr<QNNPerf>(new QNNPerf(qnnInterface));}
    QNNPerf(const QNN_INTERFACE_VER_TYPE * qnnInterface);
    ~QNNPerf();
    void setRpcLatencyAndPolling();
    void setPowerConfigBurst();
    void setPowerConfigBalanced();

private:
    const QNN_INTERFACE_VER_TYPE * mQnnInterface = nullptr;
    QnnHtpDevice_PerfInfrastructure_t mPerfInfra{};
    uint32_t mPowerConfigId;
    QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBurst{};
    QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBalanced{};
};

} // end namespace QNN
} // end namespace MNN

#endif // end QNN_PERF_HPP
