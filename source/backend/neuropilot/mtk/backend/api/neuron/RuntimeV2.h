/**
 * \file
 * RuntimeV2.
 * ---
 * NeuronRuntimeV2 API allows user to create a NeuronRuntimeV2 from the specified .DLA file.
 * Users can enqueue asynchronous inference requests into the created runtime. Or, users can issue
 * conventional synchronous requests, too.
 */

#pragma once

#include "Types.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/// IOBuffer is a descriptor describing the buffer which will be used as an inference
/// input or output. Users should zero the whole IOBuffer, then fill those fields with valid
/// data.
typedef struct IOBuffer {
    void* buffer;
    size_t length;
    int fd;      // Normal buffer must have -1 for this. Otherwise, fill with imported buffer FD
    int offset;  // Offset for imported buffer FD
    uint64_t reserved2_should_be_init_zero;
    uint64_t reserved3_should_be_init_zero;

#ifdef __cplusplus
    IOBuffer(void* buffer, size_t length, int fd, int offset = 0)
        : buffer(buffer), length(length), fd(fd), offset(offset),
          reserved2_should_be_init_zero(0),
          reserved3_should_be_init_zero(0) {}
#endif
} IOBufferType;

/// AsyncInferenceRequest represents a single inference request to be enqueued into Runtime
/// Note that all the data pointed by pointers in AsyncInferenceRequest must remain valid
/// until the inference of that request is complete.
typedef struct {
    /// A pointer to the array of input buffer descriptions. The number of elements should
    /// equal to the result of NeuronRuntimeV2_getInputNumber();
    IOBuffer* inputs;

    /// A pointer to the array of output buffer descriptions. The number of elements should
    /// equal to the result of NeuronRuntimeV2_getOutputNumber();
    IOBuffer* outputs;

    /// A callback function specified by the user for the runtime to notify inference complete.
    /// When it's called, the ID of the job just have finished and the opaque pointer in the
    /// original request will be passed back in 'job_id' and 'opaque'. The execution status
    /// is given by 'status'. A zero status indicates success. Otherwise, the inference job
    /// has failed.
    void (*finish_cb)(uint64_t job_id, void* opaque, int status);

    /// A pointer to an opaque data, which will be passed back when finish_cb is called.
    void* opaque;
} AsyncInferenceRequest;

/// SyncInferenceRequest represents a synchronous inference request to run in the Runtime.
/// The call will block until the inference finishes.
typedef struct {
    /// A pointer to the array of input buffer descriptions. The number of elements should
    /// equal to the result of NeuronRuntimeV2_getInputNumber();
    IOBuffer* inputs;

    /// A pointer to the array of output buffer descriptions. The number of elements should
    /// equal to the result of NeuronRuntimeV2_getOutputNumber();
    IOBuffer* outputs;
} SyncInferenceRequest;

/**
 * Create a NeuronRuntimeV2 based on the setting specified in options. It acts as a thread
 * pool, waiting to accept AsyncInferenceRequest or SyncInferenceRequest on a DLA file.
 * When the runtime receives a request, it enqueues the request into its backlog ring buffer,
 * and the internal load balancer will dispatch the request to the appropriate thread for
 * execution. However, there is no guarantee on the order of completion of
 * AsyncInferenceRequest. The user-specified callback should be aware of this.
 * SyncInferenceRequest, on the other hand, always block until the request finishes.
 * The address of the created runtime instance will be passed back in *runtime.
 * @param pathToDlaFile The DLA file path.
 * @param nbThreads The number of threads in the runtime. Large value for 'nbThread' could result
                    in a large memory footprint. 'nbThread' is the number of working threads and
                    each thread would maintain its own working buffer, so the total memory footprint
                    of all threads could be large.
 * @param runtime The pointer will be modified to the created NeuronRuntimeV2 instance on success.
 * @param backlog The maximum size of the backlog ring buffer. In most cases, using 2048 is enough.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_create(const char* pathToDlaFile, size_t nbThreads, void** runtime,
                           size_t backlog);

/**
 * Like NeuronRuntimeV2_create(), but it takes an additional option string.
 * @param pathToDlaFile The DLA file path.
 * @param nbThreads The number of threads in the runtime. Large value for 'nbThread' could result
                    in a large memory footprint. 'nbThread' is the number of working threads and
                    each thread would maintain its own working buffer, so the total memory footprint
                    of all threads could be large.
 * @param runtime The pointer will be modified to the created NeuronRuntimeV2 instance on success.
 * @param backlog The maximum size of the backlog ring buffer. In most cases, using 2048 is enough.
 * @param options A null-terminated C-string specifying runtime options.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_create_with_options(const char* pathToDlaFile, size_t nbThreads, void** runtime,
                                        size_t backlog, const char* options);

/**
 * Like NeuronRuntimeV2_create(), but it creates the Runtime instance from a memory buffer
 * containing the DLA data.
 * @param buffer The DLA data buffer.
 * @param len The DLA data buffer size.
 * @param nbThreads The number of threads in the runtime. Large value for 'nbThread' could result
                    in a large memory footprint. 'nbThread' is the number of working threads and
                    each thread would maintain its own working buffer, so the total memory footprint
                    of all threads could be large.
 * @param runtime The pointer will be modified to the created NeuronRuntimeV2 instance on success.
 * @param backlog The maximum size of the backlog ring buffer. In most cases, using 2048 is enough.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_createFromBuffer(const void* buffer, size_t len, size_t nbThreads,
                                     void** runtime, size_t backlog);

/**
 * Like NeuronRuntimeV2_createFromBuffer(), but it takes an additional option string.
 * containing the DLA data.
 * @param buffer The DLA data buffer.
 * @param len The DLA data buffer size.
 * @param nbThreads The number of threads in the runtime. Large value for 'nbThread' could result
                    in a large memory footprint. 'nbThread' is the number of working threads and
                    each thread would maintain its own working buffer, so the total memory footprint
                    of all threads could be large.
 * @param runtime The pointer will be modified to the created NeuronRuntimeV2 instance on success.
 * @param backlog The maximum size of the backlog ring buffer. In most cases, using 2048 is enough.
 * @param options A null-terminated C-string specifying runtime options.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_createFromBuffer_with_options(const void* buffer, size_t len, size_t nbThreads,
                                                  void** runtime, size_t backlog,
                                                  const char* options);

/**
 * Release the runtime. Calling this function will block until all requests finish.
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 */
void NeuronRuntimeV2_release(void* runtime);

/**
 * Enqueue one AsyncInferenceRequest. If the backlog ring buffer is not full, this
 * function returns immediately, and the runtime will execute the request asynchronously. If
 * the backlog is full (due to back pressure from execution), this call will block until the
 * backlog ring buffer releases at least one available slot for the request. A unique ID is
 * returned for the enqueued request in *job_id. The ID sequence starts from zero and
 * increases with each received request. The 2^64 capacity for job ID should be enough for
 * any applications.
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param request The asynchronous inference request
 * @param job_id The ID for this request is filled into *job_id. Later the ID will be passed
 *               back when the finish_cb is called.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_enqueue(void* runtime, AsyncInferenceRequest request, uint64_t* job_id);

/**
 * Perform a synchronous inference request. The request will be also enqueued into the
 * Runtime ring buffer as NeuronRuntimeV2_enqueue() does. However, the call will block until
 * the request finishes.
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param request The synchronous inference request
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_run(void* runtime, SyncInferenceRequest request);

/**
 * Get the number of inputs of the model in the runtime. The number of inputs will be passed
 * back in *size
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param size The pointer to a size_t to store the passed back value.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getInputNumber(void* runtime, size_t* size);

/**
 * Get the number of outputs of the model in the runtime. The number of outputs will be
 * passed back in *size
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param size The pointer to a size_t to store the passed back value.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getOutputNumber(void* runtime, size_t* size);

/**
 * Get the rank required by the input tensor (specified by handle).
 * Pass back the expected rank in *rank for the tensor which holds the specified input handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param rank The input rank.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getInputRank(void* runtime, uint64_t handle, uint32_t* rank);

/**
 * Get the physical size required by the buffer of the input tensor (specified by handle).
 * Pass back the expected buffer size (byte) in *size for the tensor which holds the specified
 * input handle.
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param handle The frontend IO index.
 * @param size The input buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getInputSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the physical size required by the buffer of the output tensor (specified by handle).
 * This funxtion passes back the expected buffer size (byte) in *size for the tensor which holds the
 * specified output handle.
 * @param runtime The address of the created NeuronRuntimeV2 instance.
 * @param handle The frontend IO index.
 * @param size The output buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getOutputSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the physical size required by the buffer of the input tensor (specified by handle) with
 * hardware alignments. This function passes back the expected buffer size (byte) in *size for the
 * tensor which holds the specified input handle. The value in *size has been aligned to hardware
 * required size, and it can be used as ION buffer size for the specified input when
 * suppressInputConversion is enabled.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param size The input buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getInputPaddedSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the size in pixels for each dimensions of the input tensor (specified by handle).
 * This function passes back the expected size (in pixels) of each dimensions in *dim for the tensor
 * which holds the specified input handle. The sizes of each dimensions in *dim have been aligned
 * to hardware required sizes. When suppressInputConversion is enabled, the values in *dim are the
 * required sizes of each dimensions for the specified input.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param dims The size (in pixels) of each dimensions.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getInputPaddedDimensions(void* runtime, uint64_t handle,
                                             RuntimeAPIDimensions* dims);

/**
 * Get the physical size required by the buffer of the output tensor (specified by handle) with
 * hardware alignments. This function passes back the expected buffer size (byte) in *size for the
 * tensor which holds the specified output handle. The value in *size has been aligned to hardware
 * required size, and it can be used as ION buffer size for the specified output when
 * suppressOutputConversion is enabled.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param size The output buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getOutputPaddedSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the size in pixels for each dimensions of the output tensor (specified by handle).
 * This function passes back the expected size (in pixels) of each dimensions in *dim for the tensor
 * which holds the specified output handle. The sizes of each dimensions in *dim have been aligned
 * to hardware required sizes. When suppressOutputConversion is enabled, the values in *dim are the
 * required sizes of each dimensions for the specified output.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param dims The size (in pixels) of each dimensions.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getOutputPaddedDimensions(void* runtime, uint64_t handle,
                                            RuntimeAPIDimensions* dims);

/**
 * Get the rank required by the output tensor (specified by handle).
 * Pass back the expected rank in *rank for the tensor which holds the specified output handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param rank The output rank.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getOutputRank(void* runtime, uint64_t handle, uint32_t* rank);

/**
 * Set shape for the input tensor which hold the specified input handle in the
 * original network. If there are multiple inputs with dynamic shapes, each of
 * them have to be set. This API is only used when input is dynamic shape, otherwise
 * error code will be returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param dims A array of dimension sizes for each dimension. For NHWC, dims[0] is N.
 * @param rank The input rank. For exmaple, rank is 4 for NHWC.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_setInputShape(void* runtime, uint64_t handle, uint32_t* dims, uint32_t rank);

/**
 * Set the QoS configuration for Neuron Runtime. If qosOption.profiledQoSData is not null,
 * Neuron Runtime would use it to store the profiled QoS data.
 * *** Note : qosOption.profiledQoSData has no effect at all.
 * *** Note : Using this API when NeuronRuntimeV2 is working leads to undefined behavior.
 *            Namely, this API should be used only when all requests have finished and no
 *            new request is being issued.
 * @param runtime The address of the created neuron runtime instance.
 * @param qosOption The option for QoS configuration.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_setQoSOption(void* runtime, const QoSOptions* qosOption);

/**
 * Get the profiled QoS data and executing boost value (the actual boost value during execution).
 * If *profiledQoSData is nullptr, Neuron Runtime would allocate *profiledQoSData.
 * Otherwise, Neuron Runtime would only update its fields.
 * *profiledQoSData is actually allocated as a smart pointer in Neuron Runtime instance,
 * so the lifetime of *profiledQoSData is the same as Neuron Runtime.
 * Caller should be careful about the usage of *profiledQoSData,
 * and never touch the allocated *profiledQoSData after NeuronRuntime_release.
 *
 * *** Note : Only effective when NeuronRuntimeV2 has nbThreads = 1.
 * *** Note : Using this API when NeuronRuntimeV2 is working leads to undefined behavior.
 *            Namely, this API should be used only when all requests have finished and no
 *            new request is being issued.
 * @note This function is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
 * @param runtime The address of the created neuron runtime instance.
 * @param profiledQoSData The profiled QoS raw data.
 * @param execBoostValue The executing boost value (the actual boot value set in device) based on
 *                       scheduling policy.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getProfiledQoSData(void* runtime, ProfiledQoSData** profiledQoSData,
                                       uint8_t* execBoostValue);


/**
 * Get metadata info in dla file, which is provided through compiler option --dla-metadata.
 * @param runtime The address of the created neuron runtime instance.
 * @param key The key for the target data
 * @param size The size of the target data. If there is no corresponding metadata, size is 0.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getMetadataInfo(void* runtime, const char* key, size_t* size);

/**
 * Get metadata in dla file, which is provided through compiler option --dla-metadata.
 * @param runtime The address of the created neuron runtime instance.
 * @param key The key for the target data
 * @param data The destination data buffer.
 * @param size The size to read from metadata.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntimeV2_getMetadata(void* runtime, const char* key, char* data, size_t size);

__END_DECLS
