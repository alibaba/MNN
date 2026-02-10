/**
 * \file
 * Neuron Runtime API
 * ---
 * Neuron provides some APIs to create runtime environment, parse compiled model file,
 * and do inference with a network.
 * \n The Runtime user should include this header to use Runtime API.
 * Note that some APIs that set input and output info need the user to specify the handle of
 * the input/output tensor that he/she wants to set.\n The user may
 * \n 1) Acts as ANN or TFLite, which always know the handle
 * \n 2) Run a precompiled network. The user should understand the model in the beginning.
 * \n 3) Run a precompiled network without knowing what the network look like. In this case,
 *    it is impossible for the user to do inference without taking a glance at the network
 *    IO map info. \n Otherwise, the user cannot even give a valid input with valid input shape.
 *    After the user checks the IO map, they would also acquire the handle and the corresponding
 *    shape.
 */

#pragma once

#if __has_include("Types.h")
#include "Types.h"
#else
#include "neuron/api/Types.h"
#endif

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/// This option controls if the underlying hardware should split and run a graph across homogeneous
/// devices. Note that this does not control the heterogeneous parallelism in the Runtime software.
/// \warning This option is to be deprecated in Neuron 6.0
typedef enum {
    Auto   = 0,  ///< Scheduler decide
    Single = 1,  ///< Force single MDLA
    Dual   = 2,  ///< Force multi MDLA
} MDLACoreMode;

typedef struct {
    /// Device kind can be chosen from kEnvOptNullDevice, or kEnvOptHardware.
    /// \n For hardware development, use kEnvOptHardware.
    uint32_t deviceKind;

    /// Set MDLA core option.
    /// \warning This option is no longer effective. To be removed in Neuron 6.0
    MDLACoreMode MDLACoreOption;

    /// Hint CPU backends to use \#threads for execution
    uint8_t CPUThreadNum;

    /// Set this to true to bypass preprocess and feed data in the format that the device demands.
    bool suppressInputConversion;

    /// Set this to true to bypass postprocess and retrieve raw device output.
    bool suppressOutputConversion;
} EnvOptions;

/// For unsigned char deviceKind.
const unsigned char kEnvOptNullDevice = 1 << 0;
const unsigned char kEnvOptHardware = 1 << 2;
const unsigned char kEnvOptPredictor = 1 << 3;

/**
 * @param options The environment options for the Neuron Runtime.
 * @return 1 to indicate user-specified EnvOptions use a NullDevice. Otherwise, return 0.
 */
inline int IsNullDevice(const EnvOptions* options) {
    return options->deviceKind & kEnvOptNullDevice;
}

/**
 * @param options The environment options for the Neuron Runtime.
 * @return 1 to indicate user-specified EnvOptions use real hardware. Otherwise, return 0.
 */
inline int IsHardware(const EnvOptions* options) {
    return options->deviceKind & kEnvOptHardware;
}

/**
 * Create a Neuron Runtime based on the setting specified in options. The address of the created
 * instance will be passed back in *runtime.
 * @param optionsToDeprecate The environment options for the Neuron Runtime (To be deprecated).
 * @param runtime Runtime provides API for applications to run a compiled network on specified
 *        input.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_create(const EnvOptions* optionsToDeprecate, void** runtime);


/**
 * Create a Neuron Runtime based on the setting specified in options. The address of the created
 * instance will be passed back in *runtime.
 * @param options The environment options for the Neuron Runtime.
 * @param optionsToDeprecate The environment options for the Neuron Runtime (To be deprecated).
 * @param runtime Runtime provides API for applications to run a compiled network on specified
 *        input.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_create_with_options(const char* options,
                                      const EnvOptions* optionsToDeprecate,
                                      void** runtime);

/**
 * Clone an existing, loaded Neuron Runtime. The existing Runtime must be DLA-loaded, and the DLA
 * file must exist, too. The address of the created instance (copy) will be passed by in
 * *newRuntime. Note that the constant data buffer may be shared with the clone runtime,
 * reducing memory footprint.
 * @param oldRuntime The existing, loaded Neuron Runtime.
 * @param newRuntime The Neuron Runtime clone.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_clone(void* oldRuntime, void** newRuntime);

/**
 * Load the compiled network from dla file.
 * @param runtime The address of the created neuron runtime instance.
 * @param pathToDlaFile The dla file path.
 * @return A RuntimeAPI error code. 0 indicating load network successfully.
 */
int NeuronRuntime_loadNetworkFromFile(void* runtime, const char* pathToDlaFile);

/**
 * Load the compiled network from a memory buffer.
 * @param runtime The address of the created neuron runtime instance.
 * @param buffer The memory buffer.
 * @param size The size of the buffer.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_loadNetworkFromBuffer(void* runtime, const void* buffer, size_t size);

/**
 * Set the memory buffer for the tensor which hold the specified input handle in the original
 * network. If there are multiple inputs, each of them have to be set.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param buffer The input buffer.
 * @param length The input buffer size.
 * @param attribute The buffer attribute for setting ION.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setInput(void* runtime, uint64_t handle, const void* buffer, size_t length,
                           BufferAttribute attribute);

/**
 * Set the memory buffer and offset for the tensor which hold the specified input handle in the
 * original network. If there are multiple inputs, each of them have to be set.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param buffer The input buffer.
 * @param length The input buffer size. This length doesn't include offset.
 * @param attribute The buffer attribute for setting ION.
 * @param offset The offset for ION buffer.
 * @param offset Reading ION buffer from start addr + offset.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setOffsetedInput(void* runtime, uint64_t handle, const void* buffer,
                                   size_t length, BufferAttribute attribute, size_t offset);

/**
 * If there is only one input, this function can set the buffer to the input automatically.
 * Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param buffer The input buffer.
 * @param length The input buffer size.
 * @param attribute The buffer attribute for setting ION.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setSingleInput(void* runtime, const void* buffer, size_t length,
                                 BufferAttribute attribute);

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
int NeuronRuntime_setInputShape(void* runtime, uint64_t handle, uint32_t* dims, uint32_t rank);

/**
 * Set the memory buffer for the tensor which hold the specified output handle in the original
 * network. If there are multiple outputs, each of them have to be set.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param buffer The output buffer.
 * @param length The output buffer size.
 * @param attribute The buffer attribute for setting ION.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setOutput(void* runtime, uint64_t handle, void* buffer, size_t length,
                            BufferAttribute attribute);

/**
 * Set the memory buffer and offset for the tensor which hold the specified output handle in the
 * original network. If there are multiple outputs, each of them have to be set.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param buffer The output buffer.
 * @param length The output buffer size. This length doesn't include offset.
 * @param attribute The buffer attribute for setting ION.
 * @param offset Writing ION buffer from start addr + offset.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setOffsetedOutput(void* runtime, uint64_t handle, void* buffer, size_t length,
                                    BufferAttribute attribute, size_t offset);

/**
 * If there is only one output, this function can set the buffer to the output automatically.
 * Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param buffer The output buffer.
 * @param length The output buffer size.
 * @param attribute The buffer attribute for setting ION.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setSingleOutput(void* runtime, void* buffer, size_t length,
                                  BufferAttribute attribute);

/**
 * Set the QoS configuration for Neuron Runtime.
 * If qosOption.profiledQoSData is not nullptr,
 * Neuron Runtime would use it as the profiled QoS data.
 * @param runtime The address of the created neuron runtime instance.
 * @param qosOption The option for QoS configuration.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_setQoSOption(void* runtime, const QoSOptions* qosOption);

/**
 * Get the number of inputs of the model in the runtime. The number of inputs will be passed
 * back in *size
 * @param runtime The address of the created NeuronRuntime instance.
 * @param size The pointer to a size_t to store the passed back value.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getInputNumber(void* runtime, size_t* size);

/**
 * Get the physical size required by the buffer of the input tensor (specified by handle).
 * Pass back the expected buffer size (byte) in *size for the tensor which holds the specified
 * input handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param size The input buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getInputSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the rank required by the input tensor (specified by handle).
 * Pass back the expected rank in *rank for the tensor which holds the specified input handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param rank The input rank.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getInputRank(void* runtime, uint64_t handle, uint32_t* rank);

/**
 * If there is only one input, this function can get the physical size required by the buffer of
 * input and return the expected buffer size (byte) in *size.
 * Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param size The input buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleInputSize(void* runtime, size_t* size);

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
int NeuronRuntime_getInputPaddedSize(void* runtime, uint64_t handle, size_t* size);

/**
 * If there is only one input, this function passes back the expected size (byte) of its buffer in
 * *size. The value in *size has been aligned to hardware required size, and it can be used as ION
 * buffer size for input when suppressInputConversion is enabled. Otherwise, the returned value is
 * NEURONRUNTIME_INCOMPLETE.
 * @param runtime The address of the created neuron runtime instance.
 * @param size The input buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleInputPaddedSize(void* runtime, size_t* size);

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
int NeuronRuntime_getInputPaddedDimensions(void* runtime, uint64_t handle,
                                           RuntimeAPIDimensions* dims);

/**
 * Get the size in pixels for each dimensions of the only input. This function passes back the
 * expected size (in pixels) of each dimensions in *dim. The sizes of each dimensions in *dim have
 * been aligned to hardware required sizes.
 * If suppressInputConversion is enabled, the values in *dim are the required sizes of each
 * dimensions for input. Otherwise NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param dims The size (in pixels) of each dimensions.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleInputPaddedDimensions(void* runtime, RuntimeAPIDimensions* dims);

/**
 * Get the number of outputs of the model in the runtime. The number of outputs will be
 * passed back in *size
 * @param runtime The address of the created NeuronRuntime instance.
 * @param size The pointer to a size_t to store the passed back value.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getOutputNumber(void* runtime, size_t* size);

/**
 * Get the physical size required by the buffer of the output tensor (specified by handle).
 * This function passes back the expected buffer size (byte) in *size for the tensor which holds the
 * specified output handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param size The output buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getOutputSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the physical size required by the buffer of the only output.
 * If there is only one Output, this function passes back the expected size (byte) of its buffer
 * in *size. Otherwise, NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param size The output buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleOutputSize(void* runtime, size_t* size);

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
int NeuronRuntime_getOutputPaddedSize(void* runtime, uint64_t handle, size_t* size);

/**
 * Get the physical size required by the buffer of the only output with hardware alignments.
 * If there is only one Output, this function passes back the expected size (byte) of its buffer in
 * *size. The value in *size has been aligned to hardware required size, and it can be used as ION
 * buffer size for output when suppressOutputConversion is enabled. Otherwise, the returned value is
 * NEURONRUNTIME_INCOMPLETE.
 * @param runtime The address of the created neuron runtime instance.
 * @param size The output buffer size.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleOutputPaddedSize(void* runtime, size_t* size);

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
int NeuronRuntime_getOutputPaddedDimensions(void* runtime, uint64_t handle,
                                            RuntimeAPIDimensions* dims);

/**
 * Get the size in pixels for each dimensions of the only output. If there is only one Output, this
 * function passes back the expected size (in pixels) of each dimensions in *dim. The sizes of each
 * dimensions in *dim have been aligned to hardware required sizes. If suppressOutputConversion is
 * enabled, the values in *dim are the required sizes of each dimensions for output. Otherwise,
 * NEURONRUNTIME_INCOMPLETE is returned.
 * @param runtime The address of the created neuron runtime instance.
 * @param dims The size (in pixels) of each dimensions.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getSingleOutputPaddedDimensions(void* runtime, RuntimeAPIDimensions* dims);

/**
 * Get the rank required by the output tensor (specified by handle).
 * Pass back the expected rank in *rank for the tensor which holds the specified output handle.
 * @param runtime The address of the created neuron runtime instance.
 * @param handle The frontend IO index.
 * @param rank The output rank.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getOutputRank(void* runtime, uint64_t handle, uint32_t* rank);

/**
 * Get the profiled QoS data and executing boost value (the actual boost value during execution).
 * If *profiledQoSData is nullptr, Neuron Runtime would allocate *profiledQoSData.
 * Otherwise, Neuron Runtime would only update its fields.
 * *profiledQoSData is actually allocated as a smart pointer in Neuron Runtime instance,
 * so the lifetime of *profiledQoSData is the same as Neuron Runtime.
 * Caller should be careful about the usage of *profiledQoSData,
 * and never touch the allocated *profiledQoSData after NeuronRuntime_release.
 * @note This function is not supported on MediaTek TV platforms (MT99XX/MT96XX/MT76XX/MT58XX).
 * @param runtime The address of the created neuron runtime instance.
 * @param profiledQoSData The profiled QoS raw data.
 * @param execBoostValue The executing boost value (the actual boot value set in device) based on
 *                       scheduling policy.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getProfiledQoSData(void* runtime, ProfiledQoSData** profiledQoSData,
                                     uint8_t* execBoostValue);

/**
 * Do inference.
 * @param runtime The address of the created neuron runtime instance.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_inference(void* runtime);

/**
 * Release the runtime resource.
 * @param runtime The address of the created neuron runtime instance.
 */
void NeuronRuntime_release(void* runtime);

/**
 * Get the version of Neuron runtime library.
 * @note Neuron runtime can only load DLA files generated by compiler with the same major version.
 * @param version the version of Neuron runtime library.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getVersion(NeuronVersion* version);

/**
 * Get metadata info in dla file, which is provided through compiler option --dla-metadata.
 * @param runtime The address of the created neuron runtime instance.
 * @param key The key for the target data
 * @param size The size of the target data. If there is no corresponding metadata, size is 0.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getMetadataInfo(void* runtime, const char* key, size_t* size);

/**
 * Get metadata in dla file, which is provided through compiler option --dla-metadata.
 * @param runtime The address of the created neuron runtime instance.
 * @param key The key for the target data
 * @param data The destination data buffer.
 * @param size The size to read from metadata.
 * @return A RuntimeAPI error code.
 */
int NeuronRuntime_getMetadata(void* runtime, const char* key, char* data, size_t size);

__END_DECLS
