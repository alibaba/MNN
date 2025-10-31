#pragma once

#if __has_include("Fence.h")
#include "Fence.h"
#else
#include "neuron/api/Fence.h"
#endif

__BEGIN_DECLS

/**
 * Check if the model supports fenced execution. Call this function after runtime is loaded with
 * model.
 * @note This function is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
 * @param dlaMuxer The address of the created NeuronDLAMuxer instance.
 * @param supported Non-zero value indicates that the model supports fenced execution.
 * @return An error code indicates whether the test model executes successfully.
 */
int NeuronDLAMuxer_isFenceSupported(void* dlaMuxer, uint8_t* supported);

/**
 * Do fenced-inference. The call should return without waiting for inference to finish. The caller
 * should prepare a FenceInfo structure and pass its address into this API. FenceFd in FenceInfo
 * will be set, and the caller can be signaled when inference completes (or error exit) by waiting
 * on the fence. Most importantly, after the fence is triggered, caller MUST call the callback in
 * fenceInfo so that Neuron can perform certain post-execution tasks. The final execution status
 * and inference time can be retrieved in FenceInfo after the callback is executed.
 * @note This function is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
 * @param dlaMuxer The address of the created NeuronDLAMuxer instance.
 * @param fenceInfo The struct is used to receive the fence file descriptor and the post-inference
 *                  callback in fenced execution.
 * @return A Runtime error code.
 */
int NeuronDLAMuxer_inferenceFenced(void* dlaMuxer, FenceInfo* fenceInfo);

__END_DECLS
