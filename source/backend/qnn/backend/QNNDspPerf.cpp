//
// QNNDspPerf.cpp
// MNN
//

#include "QNNDspPerf.hpp"
#include "DSP/QnnDspDevice.h"
#include <MNN/MNNDefine.h>

namespace MNN {
namespace QNN {

std::unique_ptr<QNNDspPerf> QNNDspPerf::create(
    const QNN_INTERFACE_VER_TYPE* qnnInterface) {
    std::unique_ptr<QNNDspPerf> perf(new QNNDspPerf(qnnInterface));
    if (!perf->initialize()) {
        return nullptr;
    }
    return perf;
}

QNNDspPerf::QNNDspPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface)
    : mQnnInterface(qnnInterface) {
}

bool QNNDspPerf::initialize() {
    if (mQnnInterface == nullptr ||
        mQnnInterface->deviceGetInfrastructure == nullptr) {
        MNN_PRINT(
            "MNN_QNN: DSP performance infrastructure is unavailable.\n");
        return false;
    }

    QnnDevice_Infrastructure_t deviceInfrastructure = nullptr;
    const auto status =
        mQnnInterface->deviceGetInfrastructure(&deviceInfrastructure);
    if (QNN_GET_ERROR_CODE(status) != QNN_SUCCESS ||
        deviceInfrastructure == nullptr) {
        MNN_PRINT(
            "MNN_QNN: DSP performance infrastructure query failed, "
            "error:%lu.\n",
            (unsigned long)status);
        return false;
    }

    auto* dspInfrastructure =
        static_cast<QnnDspDevice_Infrastructure_t*>(
            deviceInfrastructure);
    if (dspInfrastructure->createPowerConfigId == nullptr ||
        dspInfrastructure->destroyPowerConfigId == nullptr ||
        dspInfrastructure->setPowerConfig == nullptr) {
        MNN_PRINT(
            "MNN_QNN: DSP performance infrastructure is incomplete.\n");
        return false;
    }

    const auto createStatus =
        dspInfrastructure->createPowerConfigId(&mPowerConfigId);
    if (QNN_GET_ERROR_CODE(createStatus) != QNN_SUCCESS) {
        MNN_PRINT(
            "MNN_QNN: DSP power config creation failed, error:%lu.\n",
            (unsigned long)createStatus);
        return false;
    }

    mInfrastructure = dspInfrastructure;
    mInitialized = true;
    return true;
}

static bool setDspPowerOptions(
    QnnDspDevice_Infrastructure_t* infrastructure,
    uint32_t powerConfigId,
    const QnnDspPerfInfrastructure_PowerConfig_t** configs,
    const char* label) {
    const auto status =
        infrastructure->setPowerConfig(powerConfigId, configs);
    if (QNN_GET_ERROR_CODE(status) == QNN_SUCCESS) {
        return true;
    }
    MNN_PRINT(
        "MNN_QNN: DSP power option %s failed, error:%lu.\n",
        label, (unsigned long)status);
    return false;
}

bool QNNDspPerf::setPowerConfigBurst() {
    if (!mInitialized) {
        return false;
    }

    QnnDspPerfInfrastructure_PowerConfig_t dcvs =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
    dcvs.dcvsEnableConfig = 0;

    QnnDspPerfInfrastructure_PowerConfig_t sleepLatency =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    sleepLatency.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY;
    sleepLatency.sleepLatencyConfig = 40;

    QnnDspPerfInfrastructure_PowerConfig_t sleepDisable =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    sleepDisable.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_DISABLE;
    sleepDisable.sleepDisableConfig = 1;

    QnnDspPerfInfrastructure_PowerConfig_t powerMode =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    powerMode.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
    powerMode.dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

    QnnDspPerfInfrastructure_PowerConfig_t voltageCorner =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    voltageCorner.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
    voltageCorner.dcvsVoltageCornerTargetConfig =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnDspPerfInfrastructure_PowerConfig_t busVoltageCorner =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    busVoltageCorner.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
    busVoltageCorner.busVoltageCornerTargetConfig =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnDspPerfInfrastructure_PowerConfig_t coreVoltageCorner =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    coreVoltageCorner.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
    coreVoltageCorner.coreVoltageCornerTargetConfig =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnDspPerfInfrastructure_PowerConfig_t rpcLatency =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    rpcLatency.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcLatency.rpcControlLatencyConfig = 100;

    auto* infrastructure =
        static_cast<QnnDspDevice_Infrastructure_t*>(mInfrastructure);
    // QAIRT DSP resets an omitted power mode to power-save on every power
    // call. Keep all settings already verified on Snapdragon 865/V66 in one
    // atomic base vote so later calls cannot undo performance mode.
    const QnnDspPerfInfrastructure_PowerConfig_t* powerConfigs[] = {
        &dcvs,
        &sleepLatency,
        &sleepDisable,
        &powerMode,
        &voltageCorner,
        &rpcLatency,
        nullptr};
    bool applied = setDspPowerOptions(
        infrastructure, mPowerConfigId, powerConfigs, "burst-atomic");

    // QAIRT 2.36 exposes separate bus/core votes, but some V66 firmware
    // revisions reject them. Probe each without invalidating the supported
    // base vote; always carry performance mode because the DSP API otherwise
    // defaults that field to power-save for this call.
    const QnnDspPerfInfrastructure_PowerConfig_t* busPowerConfigs[] = {
        &powerMode, &busVoltageCorner, nullptr};
    const bool busApplied = setDspPowerOptions(
        infrastructure, mPowerConfigId, busPowerConfigs, "bus-max-optional");
    const QnnDspPerfInfrastructure_PowerConfig_t* corePowerConfigs[] = {
        &powerMode, &coreVoltageCorner, nullptr};
    const bool coreApplied = setDspPowerOptions(
        infrastructure, mPowerConfigId, corePowerConfigs,
        "core-max-optional");

    if (infrastructure->setMemoryConfig != nullptr) {
        QnnDspPerfInfrastructure_MemoryConfig_t vtcm =
            QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIG_INIT;
        vtcm.config =
            QNN_DSP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_VTCM_USAGE_FACTOR;
        vtcm.vtcmUsageConfig =
            QNN_DSP_PERF_INFRASTRUCTURE_VTCM_USE_FULL;
        const QnnDspPerfInfrastructure_MemoryConfig_t* memoryConfigs[] = {
            &vtcm, nullptr};
        const auto status = infrastructure->setMemoryConfig(memoryConfigs);
        if (QNN_GET_ERROR_CODE(status) != QNN_SUCCESS) {
            MNN_PRINT(
                "MNN_QNN: DSP full-VTCM config failed, error:%lu.\n",
                (unsigned long)status);
            applied = false;
        } else {
            MNN_PRINT("MNN_QNN: DSP full-VTCM config applied.\n");
        }
    } else {
        MNN_PRINT("MNN_QNN: DSP full-VTCM config is unavailable.\n");
        applied = false;
    }

    if (infrastructure->setThreadConfig != nullptr) {
        QnnDspPerfInfrastructure_ThreadConfig_t hvxThreads =
            QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIG_INIT;
        hvxThreads.config =
            QNN_DSP_PERF_INFRASTRUCTURE_THREAD_CONFIGOPTION_NUMBER_OF_HVX_THREADS;
        // Qualcomm's V66 cDSP exposes four 128-byte HVX hardware units.
        hvxThreads.numHvxThreads = 4U;
        const QnnDspPerfInfrastructure_ThreadConfig_t* threadConfigs[] = {
            &hvxThreads, nullptr};
        const auto status = infrastructure->setThreadConfig(threadConfigs);
        if (QNN_GET_ERROR_CODE(status) != QNN_SUCCESS) {
            MNN_PRINT(
                "MNN_QNN: DSP four-HVX-thread config failed, error:%lu.\n",
                (unsigned long)status);
            applied = false;
        } else {
            MNN_PRINT("MNN_QNN: DSP four-HVX-thread config applied.\n");
        }
    } else {
        MNN_PRINT("MNN_QNN: DSP HVX-thread config is unavailable.\n");
        applied = false;
    }

    if (applied) {
        MNN_PRINT(
            "MNN_QNN: DSP V66 extreme performance applied: atomic burst, "
            "DCVS MAX, sleep disabled, full VTCM, four HVX threads, "
            "bus-max=%d, core-max=%d.\n",
            busApplied ? 1 : 0, coreApplied ? 1 : 0);
    }
    return applied;
}

bool QNNDspPerf::setPowerConfigBalanced() {
    if (!mInitialized) {
        return false;
    }

    QnnDspPerfInfrastructure_PowerConfig_t dcvs =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
    dcvs.dcvsEnableConfig = 1;

    QnnDspPerfInfrastructure_PowerConfig_t sleepDisable =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    sleepDisable.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_DISABLE;
    sleepDisable.sleepDisableConfig = 0;

    QnnDspPerfInfrastructure_PowerConfig_t powerMode =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    powerMode.config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
    powerMode.dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;

    auto* infrastructure =
        static_cast<QnnDspDevice_Infrastructure_t*>(mInfrastructure);
    const QnnDspPerfInfrastructure_PowerConfig_t* powerConfigs[] = {
        &dcvs, &sleepDisable, &powerMode, nullptr};
    return setDspPowerOptions(
        infrastructure, mPowerConfigId, powerConfigs, "balanced-atomic");
}

QNNDspPerf::~QNNDspPerf() {
    if (!mInitialized) {
        return;
    }
    setPowerConfigBalanced();
    auto* infrastructure =
        static_cast<QnnDspDevice_Infrastructure_t*>(mInfrastructure);
    const auto status =
        infrastructure->destroyPowerConfigId(mPowerConfigId);
    if (QNN_GET_ERROR_CODE(status) != QNN_SUCCESS) {
        MNN_PRINT(
            "MNN_QNN: DSP power config destruction failed, error:%lu.\n",
            (unsigned long)status);
    }
}

} // end namespace QNN
} // end namespace MNN
