#ifndef MNNUTILS_HPP
#define MNNUTILS_HPP
#include <array>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>
typedef std::map<std::string, std::string> MNNMeta;
class MNNAllocator {
    // Empty
};

class MNNEnv {
    // Empty
};

class MNNConfig {
public:
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> pManager;
    MNN::Express::Module::Config pConfig;
};

namespace sherpa_mnn {

MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const void* data, size_t data_size, const int* shapedata,
    int shapeSize, halide_type_t type = halide_type_of<float>());

MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const int* shapedata,
    int shapeSize, halide_type_t type = halide_type_of<float>());
        
template <typename T>
MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const T* data, size_t data_size, const int* shapedata,
    int shapeSize) {
    return MNNUtilsCreateTensor(allocator, data, data_size, shapedata, shapeSize, halide_type_of<T>());
}


template <typename T>
MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const int* shapedata,
    int shapeSize) {
    return MNNUtilsCreateTensor(allocator, shapedata, shapeSize, halide_type_of<T>());
}


/**
    * Get the input names of a model.
    *
    * @param sess An onnxruntime session.
    * @param input_names. On return, it contains the input names of the model.
    * @param input_names_ptr. On return, input_names_ptr[i] contains
    *                         input_names[i].c_str()
    */
void GetInputNames(MNN::Express::Module *sess, std::vector<std::string> *input_names,
                    std::vector<const char *> *input_names_ptr);

/**
    * Get the output names of a model.
    *
    * @param sess An onnxruntime session.
    * @param output_names. On return, it contains the output names of the model.
    * @param output_names_ptr. On return, output_names_ptr[i] contains
    *                         output_names[i].c_str()
    */
void GetOutputNames(MNN::Express::Module *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr);

/**
    * Get the output frame of Encoder
    *
    * @param allocator allocator of onnxruntime
    * @param encoder_out encoder out tensor
    * @param t frame_index
    *
    */
MNN::Express::VARP GetEncoderOutFrame(MNNAllocator *allocator, MNN::Express::VARP encoder_out,
                                int32_t t);

std::string LookupCustomModelMetaData(const MNNMeta &meta_data,
                                        const char *key, MNNAllocator *allocator);

void PrintModelMetadata(std::ostream &os,
                        const MNNMeta &meta_data);  // NOLINT

// Return a deep copy of v
MNN::Express::VARP Clone(MNNAllocator *allocator, MNN::Express::VARP v);

// Return a shallow copy
MNN::Express::VARP View(MNN::Express::VARP v);

float ComputeSum(MNN::Express::VARP v, int32_t n = -1);
float ComputeMean(MNN::Express::VARP v, int32_t n = -1);

// Print a 1-D tensor to stderr
template <typename T = float>
void Print1D(MNN::Express::VARP v);

// Print a 2-D tensor to stderr
template <typename T = float>
void Print2D(MNN::Express::VARP v);

// Print a 3-D tensor to stderr
void Print3D(MNN::Express::VARP v);

// Print a 4-D tensor to stderr
void Print4D(MNN::Express::VARP v);

void PrintShape(MNN::Express::VARP v);

template <typename T = float>
void Fill(MNN::Express::VARP tensor, T value) {
    auto n = tensor->getInfo()->size;
    auto p = tensor->writeMap<T>();
    std::fill(p, p + n, value);
}

// TODO(fangjun): Document it
MNN::Express::VARP Repeat(MNNAllocator *allocator, MNN::Express::VARP cur_encoder_out,
                    const std::vector<int32_t> &hyps_num_split);

struct CopyableOrtValue {
    MNN::Express::VARP value{nullptr};

    CopyableOrtValue() = default;

    /*explicit*/ CopyableOrtValue(MNN::Express::VARP v)  // NOLINT
        : value(std::move(v)) {}

    CopyableOrtValue(const CopyableOrtValue &other);

    CopyableOrtValue &operator=(const CopyableOrtValue &other);

    CopyableOrtValue(CopyableOrtValue &&other) noexcept;

    CopyableOrtValue &operator=(CopyableOrtValue &&other) noexcept;
};

std::vector<CopyableOrtValue> Convert(std::vector<MNN::Express::VARP> values);

std::vector<MNN::Express::VARP> Convert(std::vector<CopyableOrtValue> values);
    

};
#endif
