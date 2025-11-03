/**
 * \file
 * Types.h
 * ---
 * Common type definitions.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/// Execution preference.
/// @note This enum is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
typedef enum {
    /// Prefer performance.
    NEURONRUNTIME_PREFER_PERFORMANCE = 0,
    /// Prefer low power.
    NEURONRUNTIME_PREFER_POWER,
    /// Hint for turbo boost mode.
    /// Only valid for certain platforms (e.g., DX-1),
    /// For other platforms without turbo boost mode support, the behavior of
    /// NEURONRUNTIME_HINT_TURBO_BOOST would be identical to NEURONRUNTIME_PREFER_PERFORMANCE.
    NEURONRUNTIME_HINT_TURBO_BOOST,
} RuntimeAPIQoSPreference;

/// Task priority.
typedef enum {
    NEURONRUNTIME_PRIORITY_LOW = 0,  ///< Low priority.
    NEURONRUNTIME_PRIORITY_MED,      ///< Medium priority.
    NEURONRUNTIME_PRIORITY_HIGH,     ///< High priority.
} RuntimeAPIQoSPriority;

/// Special boost value hint.
/// @note This enum is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
typedef enum {
    /// 101: Hint to notify the scheduler to use the profiled boost value.
    NEURONRUNTIME_BOOSTVALUE_PROFILED = 101,
    NEURONRUNTIME_BOOSTVALUE_MAX = 100,  ///< 100: Maximum boost value
    NEURONRUNTIME_BOOSTVALUE_MIN = 0,    ///< 0: Minimum boost value
} RuntimeAPIQoSBoostValue;

/// Delayed power off time.
/// @note This enum is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
typedef enum {
    /// Default power off time
    NEURONRUNTIME_POWER_OFF_TIME_DEFAULT = -1,
} RuntimeAPIQoSDelayedPowerOffTime;

/// Power policy.
/// @note This enum is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
typedef enum {
    /// Default policy.
    NEURONRUNTIME_POWER_POLICY_DEFAULT = 0,
    NEURONRUNTIME_POWER_POLICY_SUSTAINABLE = 1,
    NEURONRUNTIME_POWER_POLICY_PERFORMANCE = 2,
    NEURONRUNTIME_POWER_POLICY_POWER_SAVING = 3,
} RuntimeAPIQoSPowerPolicy;

/// Application type.
/// @note This enum is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
typedef enum {
    /// Normal type.
    NEURONRUNTIME_APP_NORMAL = 0,
} RuntimeAPIQoSAppType;

/**
 * Raw data for QoS configuration.
 * All of those fields should be filled with the profiled data.
 * @note This struct is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
 */
typedef struct {
    /// <b> Profiled execution time </b> :
    /// the profiled execution time (in usec).
    uint64_t execTime;
    /// <b> Suggested time </b>:
    /// the suggested time (in msec).
    uint32_t suggestedTime;
    /// <b> Profled bandwidh </b>:
    /// the profiled bandwidh (in MB/s).
    uint32_t bandwidth;
    /// <b> Profiled boost value </b>:
    /// the profiled executing boost value (range in 0 to 100).
    uint8_t boostValue;
} QoSData;

/// Maintain the profiled QoS raw data.
typedef struct {
    /// Maintain profiled QoS raw data in a pointer of pointer.\n
    /// This field could be nullptr if there is no previous profiled data.
    QoSData** qosData;
    /// Number of sub-command in *qosData.\n
    /// This field could be nullptr if there is no previous profiled data.
    uint32_t* numSubCmd;
    /// Number of subgraph.\n
    /// This field should be zero if there is no previous profiled data.
    uint32_t numSubgraph;
} ProfiledQoSData;

/// QoS Option for configuration.
typedef struct {
    /// <b>Execution preference</b>:\n
    /// NEURONRUNTIME_PREFER_PERFORMANCE, NEURONRUNTIME_PREFER_POWER,
    /// or NEURONRUNTIME_HINT_TURBO_BOOST.
    RuntimeAPIQoSPreference preference;
    /// <b>Task priority</b>: \n NEURONRUNTIME_PRIORITY_HIGH, NEURONRUNTIME_PRIORITY_MED,
    /// or NEURONRUNTIME_PRIORITY_LOW.
    RuntimeAPIQoSPriority priority;
    /// <b>Boost value hint</b>: hint for the device frequency, ranged between 0 (lowest) to 100
    /// (highest). This value is the hint for baseline boost value in the scheduler,
    /// which sets the executing boost value (the actual boot value set in device) based on
    /// scheduling policy.
    /// For the inferences with preference set as NEURONRUNTIME_PREFER_PERFORMANCE, scheduler
    /// guarantees that the executing boost value would not be lower than the boost value hint.
    /// On the other hand, for the inferences with preference set as NEURONRUNTIME_PREFER_POWER,
    /// scheduler would try to save power by configuring the executing boost value with some value
    /// that is not higher than the boost value hint.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    uint8_t boostValue;
    /// <b>Maximum boost value</b>: reserved.
    /// Assign 0 to this field by default.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    uint8_t maxBoostValue;
    /// <b>Minimum boost value</b>: reserved.
    /// Assign 0 to this field by default.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    uint8_t minBoostValue;
    /// \b Deadline: deadline for the inference (in msec).
    /// Setting any non-zero value would nofity the scheduler that this inference is
    /// a real-time task.
    /// This field should be zero, unless this inference is a real-time task.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    uint16_t deadline;
    /// <b>Abort time</b>: the maximum inference time for the inference (in msec).
    /// If the inference is not completed before the abort time, the scheduler would
    /// abort the inference.
    /// This field should be zero, unless you wish to abort the inference.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    uint16_t abortTime;
    /// <b>Delayed power off time</b>: delayed power off time after inference completed (in msec).
    /// Scheduler would start a timer for the time interval defined in delayed power off time
    /// after the inference completion. Once the delayed power off time expired and there is no
    /// other incoming inference requests, the underlying devices would be powered off for
    /// power-saving purpose.
    /// Set this field to NEURONRUNTIME_POWER_OFF_TIME_DEFAULT to use the default power off policy
    /// in the scheduler.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    int32_t delayedPowerOffTime;
    /// <b>Power policy</b>: configure power policy for scheduler.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    RuntimeAPIQoSPowerPolicy powerPolicy;
    /// <b>Application type</b>: hint for the application type for the inference.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    RuntimeAPIQoSAppType applicationType;
    /// <b>Profiled QoS Data</b>: pointer to the historical QoS data of previous inferences.
    /// If there is no profiled data, this field could be nullptr.
    /// For the details, please check the ProfiledQoSData part.
    /// @note This member is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
    ProfiledQoSData* profiledQoSData;
} QoSOptions;


// The dimension size of RuntimeAPIDimensions.
const uint32_t kDimensionSize = 4;

/// The aligned sizes of dimensions. For NHWC format, dimensions[0] is N.
typedef struct {
    uint32_t dimensions[kDimensionSize];
} RuntimeAPIDimensions;

/// The structure to represent the neuron version.
typedef struct {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} NeuronVersion;

/// A Neuron Runtime API returns an error code to show the status of execution.
typedef enum {
    NEURONRUNTIME_NO_ERROR        = 0,  ///< 0: The API is complete successfully.
    NEURONRUNTIME_OUT_OF_MEMORY   = 1,  ///< 1: Memory is not enough for the API.
    NEURONRUNTIME_INCOMPLETE      = 2,  ///< 2: Not in use.
    NEURONRUNTIME_UNEXPECTED_NULL = 3,  ///< 3: A required pointer is null.
    NEURONRUNTIME_BAD_DATA        = 4,  ///< 4: Failed to load data or set input/output.
    NEURONRUNTIME_BAD_STATE       = 5,  ///< 5: Not in use.
    NEURONRUNTIME_RUNTIME_ERROR   = 6,  ///< 6: Hardware or simulator return unexpectedly.
} RuntimeAPIErrorCode;

/// BufferAttribute is used to inform the runtime whether this buffer is an ION buffer. If ionFd is
/// -1, the buffer is a non-ION buffer. Otherwise, the buffer is an ION buffer and ionFd is its
/// shared ION buffer file descriptor. Android device implementations may benefit from this
/// information to eliminate unnecessary data copy.
typedef struct {
    int ionFd;  ///< -1: Non-ION buffer.
} BufferAttribute;

#define NON_ION_FD -1

__END_DECLS
