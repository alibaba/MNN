//
//  CoreMLHelper.hpp
//  MNN
//
//  Created by MNN on 2025/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLHELPER_H
#define MNN_COREMLHELPER_H
#include <MNN/Tensor.hpp>
#include <cstdint>
#include <map>
#include <memory> // for std::unique_ptr
#include <string>
#include <type_traits> // for std::is_same
#include <vector>
#include "Model.pb-c.h"

namespace MNN {
class Tensor;
// --- Helper to allocate and initialize protobuf-c structs ---
template <typename T>
T* create() {
    auto t = new T;
    return t;
}

template <typename T>
T* create(size_t n) {
    return (T*)malloc(sizeof(T) * n);
}

template <typename T>
T* create_and_init(); // Forward declaration for specific types
char* s_strdup(const std::string& str);

// Specializations for each protobuf type
#define DECLARE_CREATE_AND_INIT(ProtoType, InitFunc) \
    template <>                                      \
    ProtoType* create_and_init<ProtoType>();

DECLARE_CREATE_AND_INIT(CoreML__Specification__Model, core_ml__specification__model__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__ModelDescription, core_ml__specification__model_description__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__FeatureDescription, core_ml__specification__feature_description__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__FeatureType, core_ml__specification__feature_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__StateFeatureType, core_ml__specification__state_feature_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__ArrayFeatureType, core_ml__specification__array_feature_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__ArrayFeatureType__ShapeRange,
                        core_ml__specification__array_feature_type__shape_range__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__SizeRange, core_ml__specification__size_range__init)

// MILSpec types
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program, core_ml__specification__milspec__program__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program__FunctionsEntry,
                        core_ml__specification__milspec__program__functions_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function, core_ml__specification__milspec__function__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__NamedValueType,
                        core_ml__specification__milspec__named_value_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__ValueType, core_ml__specification__milspec__value_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorType, core_ml__specification__milspec__tensor_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__StateType, core_ml__specification__milspec__state_type__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension, core_ml__specification__milspec__dimension__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension__ConstantDimension,
                        core_ml__specification__milspec__dimension__constant_dimension__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function__BlockSpecializationsEntry,
                        core_ml__specification__milspec__function__block_specializations_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Block, core_ml__specification__milspec__block__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation, core_ml__specification__milspec__operation__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation__InputsEntry,
                        core_ml__specification__milspec__operation__inputs_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Argument, core_ml__specification__milspec__argument__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue,
                        core_ml__specification__milspec__tensor_value__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program__AttributesEntry,
                        core_ml__specification__milspec__program__attributes_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__DictionaryValue,
                        core_ml__specification__milspec__dictionary_value__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__DictionaryValue__KeyValuePair,
                        core_ml__specification__milspec__dictionary_value__key_value_pair__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value, core_ml__specification__milspec__value__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function__AttributesEntry,
                        core_ml__specification__milspec__function__attributes_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Block__AttributesEntry,
                        core_ml__specification__milspec__block__attributes_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation__AttributesEntry,
                        core_ml__specification__milspec__operation__attributes_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorType__AttributesEntry,
                        core_ml__specification__milspec__tensor_type__attributes_entry__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Argument__Binding,
                        core_ml__specification__milspec__argument__binding__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value__ImmediateValue,
                        core_ml__specification__milspec__value__immediate_value__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension__UnknownDimension,
                        core_ml__specification__milspec__dimension__unknown_dimension__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedFloats,
                        core_ml__specification__milspec__tensor_value__repeated_floats__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedInts,
                        core_ml__specification__milspec__tensor_value__repeated_ints__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedStrings,
                        core_ml__specification__milspec__tensor_value__repeated_strings__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedBools,
                        core_ml__specification__milspec__tensor_value__repeated_bools__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedBytes,
                        core_ml__specification__milspec__tensor_value__repeated_bytes__init)
DECLARE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value__BlobFileValue,
                        core_ml__specification__milspec__value__blob_file_value__init)

#undef DECLARE_CREATE_AND_INIT


template <typename T>
struct MilTypeTraits {
    static_assert(sizeof(T) == 0, "Unsupported type for buildScalar");
};

template <>
struct MilTypeTraits<int32_t> {
    static constexpr auto dataType = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32;
};

template <>
struct MilTypeTraits<float> {
    static constexpr auto dataType = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
};

template <>
struct MilTypeTraits<std::string> {
    static constexpr auto dataType = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__STRING;
};

template <>
struct MilTypeTraits<bool> {
    static constexpr auto dataType = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL;
};

struct MilTensorInfo {
    enum class Type { TENSOR, INPUT, OUTPUT, STATE };
    std::string name;
    CoreML__Specification__MILSpec__DataType data_type;
    std::vector<int64_t> shape;
    Type type;
    void update(MilTensorInfo* other);
};

class MilValueBuilder {
public:
    enum class Type {
        BLOB_FILE,
        ARRAY_SHAPE,
        ARRAY_INT32,
        ARRAY_FLOAT,
        ARRAY_BOOL,
        SCALAR_INT32,
        SCALAR_FLOAT,
        SCALAR_FLOAT32,
        SCALAR_STRING,
        SCALAR_BOOL
    };
    static std::unique_ptr<MilValueBuilder> createMilShapeBuilder(const std::vector<int>& val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(const std::vector<int>& val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(const std::vector<float>& val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(const std::vector<bool>& val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(int val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(float val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(const std::string& val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(bool val);
    static std::unique_ptr<MilValueBuilder> createMilValueBuilder(const std::string& file,
                                                                  CoreML__Specification__MILSpec__DataType data_type,
                                                                  const std::vector<int>& shape, uint64_t offset);
    CoreML__Specification__MILSpec__Value* build();

public:
    Type type;
    std::vector<int32_t> array_int32;
    std::vector<float> array_float;
    std::vector<bool> array_bool;
    int32_t scalar_int32;
    float scalar_float;
    std::string scalar_string;
    bool scalar_bool;
    uint64_t scalar_uint64;
    CoreML__Specification__MILSpec__DataType data_type;
};

// --- MilOperationBuilder ---
class MilOperationBuilder {
public:
    MilOperationBuilder();
    MilOperationBuilder(const std::string& type);
    // Non-copyable
    MilOperationBuilder(const MilOperationBuilder&)            = delete;
    MilOperationBuilder& operator=(const MilOperationBuilder&) = delete;
    // Movable
    MilOperationBuilder(MilOperationBuilder&& other)            = delete;
    MilOperationBuilder& operator=(MilOperationBuilder&& other) = delete;
    ~MilOperationBuilder();
    void setType(const std::string& type);
    void setName(const std::string& name);
    std::string getType();
    std::string getName();
    void addInput(const std::string& param_name, const std::string& tensor_name);
    void addInputs(const std::string& param_name, const std::vector<std::string>& tensor_names);
    void addOutput(const std::string& name, CoreML__Specification__MILSpec__DataType data_type,
                   const std::vector<int64_t>& shape);
    void addAttribute(const std::string& attr_name, std::unique_ptr<MilValueBuilder> attr_value);
    bool usesInput(const std::string& name) const;
    void update(MilOperationBuilder* other);
    std::unique_ptr<MilOperationBuilder> clone(const std::map<std::string, std::string>& names,
                                               const std::string& prefix) const;
    CoreML__Specification__MILSpec__Operation* build();

private:
    std::string mType, mName;
    std::map<std::string, std::vector<std::string>> mInputs;
    std::vector<std::unique_ptr<MilTensorInfo>> mOutputs;
    std::map<std::string, std::unique_ptr<MilValueBuilder>> mAttributes;
};

// -- MlModelBuilder ---
class MlModelBuilder {
public:
    MlModelBuilder(int32_t version = 8);
    // Non-copyable
    MlModelBuilder(const MlModelBuilder&)             = delete;
    MlModelBuilder& operator=(const MlModelBuilder&)  = delete;
    MlModelBuilder(MlModelBuilder&& other)            = delete;
    MlModelBuilder& operator=(MlModelBuilder&& other) = delete;
    ~MlModelBuilder();
    // input and output description
    void addInput(std::unique_ptr<MilTensorInfo> info);
    void addOutput(std::unique_ptr<MilTensorInfo> info);
    // operations
    void addOperation(std::unique_ptr<MilOperationBuilder> builer);
    bool usesInput(const std::string& name) const;
    void appendOperations(const MlModelBuilder& source, const std::map<std::string, std::string>& names,
                          const std::string& prefix);
    size_t numOperations() const {
        return mOperationBuilders.size();
    }
    // Keep constructor-time constants, rebuild shape-dependent operations on resize.
    void markStaticOperations();
    void resetOperations();
    // Build operations
    CoreML__Specification__Model* build();

private:
    CoreML__Specification__MILSpec__Block* buildBlock();
    CoreML__Specification__MILSpec__Function* buildFunction();
    CoreML__Specification__MILSpec__Program* buildProgram();
    CoreML__Specification__FeatureDescription* buildFeatureDescription(const MilTensorInfo* info);

private:
    int mVersion                         = 8;
    CoreML__Specification__Model* mModel = nullptr;
    std::vector<std::unique_ptr<MilOperationBuilder>> mOperationBuilders;
    std::vector<std::unique_ptr<MilTensorInfo>> mInputs, mOutputs;
    std::map<std::string, MilTensorInfo*> mIOMap;
    std::map<std::string, MilOperationBuilder*> mOperationBuilderMap;
    size_t mStaticOperationCount = 0;
};

// --- MilWeightWriter ---
enum class BlobDataType : uint32_t {
    Float16      = 1,
    Float32      = 2,
    UInt8        = 3,
    Int8         = 4,
    BFloat16     = 5,
    Int16        = 6,
    UInt16       = 7,
    Int4         = 8,
    UInt1        = 9,
    UInt2        = 10,
    UInt4        = 11,
    UInt3        = 12,
    UInt6        = 13,
    Int32        = 14,
    UInt32       = 15,
    Float8E4M3FN = 16,
    Float8E5M2   = 17,
};

class MilWeightWriter {
public:
    explicit MilWeightWriter(const std::string& filePath);
    MilWeightWriter(const std::shared_ptr<std::vector<uint8_t>>& data, size_t capacity);
    ~MilWeightWriter();
    uint64_t write(const void* data, size_t size, CoreML__Specification__MILSpec__DataType dtype);
    uint64_t tell() const;
    void writeRaw(const void* p, size_t sz);
    void flushHeader();
    void alignFile();
    bool valid() const { return mValid; }

private:
    struct storage_header {
        uint32_t count      = 0; // Number of blob data.
        uint32_t version    = 2; // default=2
        uint64_t reserved_0 = 0;
        uint64_t reserved_1 = 0;
        uint64_t reserved_2 = 0;
        uint64_t reserved_3 = 0;
        uint64_t reserved_4 = 0;
        uint64_t reserved_5 = 0;
        uint64_t reserved_6 = 0;
    };
    FILE* mFile = nullptr;
    std::shared_ptr<std::vector<uint8_t>> mData;
    std::string mFilePath;
    bool mValid = true;
    storage_header mHeader;
};

// --- Helper for TensorType (MIL) ---
CoreML__Specification__MILSpec__TensorType* createMilTensorType(CoreML__Specification__MILSpec__DataType data_type,
                                                                const std::vector<int64_t>& shape_values);
// --- Helper for ValueType (MIL) ---
CoreML__Specification__MILSpec__ValueType* createMilValueTypeTensor(CoreML__Specification__MILSpec__DataType data_type,
                                                                    const std::vector<int64_t>& shape_values);
CoreML__Specification__MILSpec__ValueType* createMilValueTypeState(CoreML__Specification__MILSpec__DataType data_type,
                                                                    const std::vector<int64_t>& shape_values);
// --- Helper for NamedValueType (MIL) ---
CoreML__Specification__MILSpec__NamedValueType* createMilNamedValueType(const std::string& name,
                                            CoreML__Specification__MILSpec__ValueType* value_type);

} // namespace MNN
#endif // MNN_COREMLHELPER_H
