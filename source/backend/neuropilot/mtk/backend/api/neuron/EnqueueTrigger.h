#pragma once

#if __has_include("Fence.h")
#include "Fence.h"
#else
#include "neuron/api/Fence.h"
#endif

#include <stdint.h>

__BEGIN_DECLS

// Introduction to Enqueue Trigger mechanism.
// Enqueue Trigger is an inference mechanism that can separate inference into two stage,
// (1) enqueue-stage and (2) trigger-stage.
// All the pre-execution tasks would be done in enqueue-stage. Then in trigger-stage we
// will only trigger the enqueued job to execute.
// Once the inference settings are changed. User should enqueue the job one more time.
// Then after that user can trigger that job any time they want before session released.

/**
 * Check if the model supports enqueueThenTrigger execution. Call this function after
 * runtime is loaded with model.
 * @param runtime The address of the created neuron runtime instance.
 * @param supported Non-zero value indicates that the model supports enqueueThenTrigger
 * execution.
 * @return An error code indicates whether the test model executes successfully.
 */
int NeuronRuntime_isEnqueueTriggerSupported(void* runtime, uint8_t* supported);

/**
 * Do job-enqueue. All the pre-execution task will be done in this API.
 * This API will enqueue the job then waiting for the trigger signals.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronRuntime_inferenceEnqueue(void* runtime);

/**
 * Do job-trigger. Trigger job that user enqueued before. It is expected that
 * NeuronRuntime_inferenceEnqueue() has been called before and it is the last API
 * invoked before calling this API.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronRuntime_inferenceTrigger(void* runtime);

/**
 * Do job-trigger-fenced. Trigger job that user enqueued before. It is expected that
 * NeuronRuntime_inferenceEnqueue() has been called before and it is the last API
 * invoked before calling this API.
 * The call should return without waiting for inference to finish. The caller should
 * prepare a FenceInfo structure and pass its address into this API. FenceFd in FenceInfo
 * will be set and the caller can be signaled when inference completes (or error exit) by
 * waiting on the fence. Most importantly, after the fence is triggered, caller MUST call
 * the callback in fenceInfo so that Neuron can perform certain post-execution tasks. The
 * final execution status and inference time can be retrieved in FenceInfo after the
 * callback is executed.
 * @param runtime The address of the created neuron runtime instance.
 * @return A Runtime error code.
 */
int NeuronRuntime_inferenceTriggerFenced(void* runtime, FenceInfo* fenceInfo);

__END_DECLS
