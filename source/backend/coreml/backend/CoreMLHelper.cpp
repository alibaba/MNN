// CoreMLHelper.cpp
// MNN
#include "CoreMLHelper.hpp"
#include <cstring>
#include "backend/arm82/Arm82OptFunc.hpp"

namespace MNN {

#define DEFINE_CREATE_AND_INIT(ProtoType, InitFunc)             \
    template <>                                                 \
    ProtoType* create_and_init<ProtoType>() {                   \
        ProtoType* ptr = (ProtoType*)malloc(sizeof(ProtoType)); \
        InitFunc(ptr);                                          \
        return ptr;                                             \
    }

DEFINE_CREATE_AND_INIT(CoreML__Specification__Model, core_ml__specification__model__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__ModelDescription, core_ml__specification__model_description__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__FeatureDescription, core_ml__specification__feature_description__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__FeatureType, core_ml__specification__feature_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__StateFeatureType, core_ml__specification__state_feature_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__ArrayFeatureType, core_ml__specification__array_feature_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__ArrayFeatureType__ShapeRange,
                       core_ml__specification__array_feature_type__shape_range__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__SizeRange, core_ml__specification__size_range__init)
// MILSpec types
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program, core_ml__specification__milspec__program__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program__FunctionsEntry,
                       core_ml__specification__milspec__program__functions_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function, core_ml__specification__milspec__function__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__NamedValueType,
                       core_ml__specification__milspec__named_value_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__ValueType, core_ml__specification__milspec__value_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorType, core_ml__specification__milspec__tensor_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__StateType, core_ml__specification__milspec__state_type__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension, core_ml__specification__milspec__dimension__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension__ConstantDimension,
                       core_ml__specification__milspec__dimension__constant_dimension__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function__BlockSpecializationsEntry,
                       core_ml__specification__milspec__function__block_specializations_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Block, core_ml__specification__milspec__block__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation, core_ml__specification__milspec__operation__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation__InputsEntry,
                       core_ml__specification__milspec__operation__inputs_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Argument, core_ml__specification__milspec__argument__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue, core_ml__specification__milspec__tensor_value__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Program__AttributesEntry,
                       core_ml__specification__milspec__program__attributes_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__DictionaryValue,
                       core_ml__specification__milspec__dictionary_value__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__DictionaryValue__KeyValuePair,
                       core_ml__specification__milspec__dictionary_value__key_value_pair__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value, core_ml__specification__milspec__value__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Function__AttributesEntry,
                       core_ml__specification__milspec__function__attributes_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Block__AttributesEntry,
                       core_ml__specification__milspec__block__attributes_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Operation__AttributesEntry,
                       core_ml__specification__milspec__operation__attributes_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorType__AttributesEntry,
                       core_ml__specification__milspec__tensor_type__attributes_entry__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Argument__Binding,
                       core_ml__specification__milspec__argument__binding__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value__ImmediateValue,
                       core_ml__specification__milspec__value__immediate_value__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Dimension__UnknownDimension,
                       core_ml__specification__milspec__dimension__unknown_dimension__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedFloats,
                       core_ml__specification__milspec__tensor_value__repeated_floats__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedInts,
                       core_ml__specification__milspec__tensor_value__repeated_ints__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedStrings,
                       core_ml__specification__milspec__tensor_value__repeated_strings__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedBools,
                       core_ml__specification__milspec__tensor_value__repeated_bools__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__TensorValue__RepeatedBytes,
                       core_ml__specification__milspec__tensor_value__repeated_bytes__init)
DEFINE_CREATE_AND_INIT(CoreML__Specification__MILSpec__Value__BlobFileValue,
                       core_ml__specification__milspec__value__blob_file_value__init)
#undef DEFINE_CREATE_AND_INIT

char* s_strdup(const std::string& str) {
    if (str.empty()) {
        return nullptr;
    }
    char* cstr = strdup(str.c_str());
    return cstr;
}

// --- Helper for TensorType (MIL) ---
CoreML__Specification__MILSpec__TensorType* createMilTensorType(CoreML__Specification__MILSpec__DataType data_type,
                                                                const std::vector<int64_t>& shape_values) {
    auto tensor_type      = create_and_init<CoreML__Specification__MILSpec__TensorType>();
    tensor_type->datatype = data_type;
    tensor_type->rank     = shape_values.size();
    if (!shape_values.empty()) {
        tensor_type->n_dimensions = shape_values.size();
        tensor_type->dimensions   = create<CoreML__Specification__MILSpec__Dimension*>(shape_values.size());
        for (size_t i = 0; i < shape_values.size(); ++i) {
            tensor_type->dimensions[i] = create_and_init<CoreML__Specification__MILSpec__Dimension>();
            if (shape_values[i] < 0) {
                tensor_type->dimensions[i]->dimension_case =
                    CORE_ML__SPECIFICATION__MILSPEC__DIMENSION__DIMENSION_UNKNOWN;
                tensor_type->dimensions[i]->unknown =
                    create_and_init<CoreML__Specification__MILSpec__Dimension__UnknownDimension>();
                tensor_type->dimensions[i]->unknown->variadic = false;
            } else {
                tensor_type->dimensions[i]->dimension_case =
                    CORE_ML__SPECIFICATION__MILSPEC__DIMENSION__DIMENSION_CONSTANT;
                tensor_type->dimensions[i]->constant =
                    create_and_init<CoreML__Specification__MILSpec__Dimension__ConstantDimension>();
                tensor_type->dimensions[i]->constant->size = shape_values[i];
            }
        }
    } else {
        tensor_type->n_dimensions = 0;
        tensor_type->dimensions   = nullptr;
    }
    return tensor_type;
}

// --- Helper for ValueType (MIL) ---
CoreML__Specification__MILSpec__ValueType* createMilValueTypeTensor(CoreML__Specification__MILSpec__DataType data_type,
                                                                    const std::vector<int64_t>& shape_values) {
    auto value_type        = create_and_init<CoreML__Specification__MILSpec__ValueType>();
    value_type->type_case  = CORE_ML__SPECIFICATION__MILSPEC__VALUE_TYPE__TYPE_TENSOR_TYPE;
    value_type->tensortype = createMilTensorType(data_type, shape_values);
    return value_type;
}
CoreML__Specification__MILSpec__ValueType* createMilValueTypeState(CoreML__Specification__MILSpec__DataType data_type,
                                                                   const std::vector<int64_t>& shape_values) {
    auto value_type                    = create_and_init<CoreML__Specification__MILSpec__ValueType>();
    value_type->type_case              = CORE_ML__SPECIFICATION__MILSPEC__VALUE_TYPE__TYPE_STATE_TYPE;
    value_type->statetype              = create_and_init<CoreML__Specification__MILSpec__StateType>();
    value_type->statetype->wrappedtype = createMilValueTypeTensor(data_type, shape_values);
    return value_type;
}
// --- Helper for NamedValueType (MIL) ---
CoreML__Specification__MILSpec__NamedValueType* createMilNamedValueType(
    const std::string& name, CoreML__Specification__MILSpec__ValueType* value_type) {
    auto named_value  = create_and_init<CoreML__Specification__MILSpec__NamedValueType>();
    named_value->name = s_strdup(name);
    named_value->type = value_type;
    return named_value;
}

template <typename T>
void updateDynamicShape(std::vector<T>& shape, const std::vector<T>& newShape) {
    if (shape.size() != newShape.size()) {
        return;
    }
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] != newShape[i]) {
            shape[i] = -1;
        }
    }
}

void MilTensorInfo::update(MilTensorInfo* other) {
    updateDynamicShape(shape, other->shape);
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilShapeBuilder(const std::vector<int>& val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type        = MilValueBuilder::Type::ARRAY_SHAPE;
    value->array_int32 = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(const std::vector<int>& val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type        = MilValueBuilder::Type::ARRAY_INT32;
    value->array_int32 = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(const std::vector<float>& val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type        = MilValueBuilder::Type::ARRAY_FLOAT;
    value->array_float = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(const std::vector<bool>& val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type       = MilValueBuilder::Type::ARRAY_BOOL;
    value->array_bool = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(int val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type         = MilValueBuilder::Type::SCALAR_INT32;
    value->scalar_int32 = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(float val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type         = MilValueBuilder::Type::SCALAR_FLOAT;
    value->scalar_float = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(const std::string& val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type          = MilValueBuilder::Type::SCALAR_STRING;
    value->scalar_string = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(bool val) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type        = MilValueBuilder::Type::SCALAR_BOOL;
    value->scalar_bool = val;
    return value;
}

std::unique_ptr<MilValueBuilder> MilValueBuilder::createMilValueBuilder(
    const std::string& file, CoreML__Specification__MILSpec__DataType data_type, const std::vector<int>& shape,
    uint64_t offset) {
    std::unique_ptr<MilValueBuilder> value(new MilValueBuilder);
    value->type          = MilValueBuilder::Type::BLOB_FILE;
    value->scalar_string = file;
    value->data_type     = data_type;
    value->array_int32   = shape;
    value->scalar_uint64 = offset;
    return value;
}

CoreML__Specification__MILSpec__Value* MilValueBuilder::build() {
    auto val = create_and_init<CoreML__Specification__MILSpec__Value>();
    switch (type) {
        case MilValueBuilder::Type::BLOB_FILE: {
            std::vector<int64_t> shape(array_int32.begin(), array_int32.end());
            val->type                    = createMilValueTypeTensor(data_type, shape);
            val->value_case              = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_BLOB_FILE_VALUE;
            val->blobfilevalue           = create_and_init<CoreML__Specification__MILSpec__Value__BlobFileValue>();
            val->blobfilevalue->filename = s_strdup(scalar_string);
            val->blobfilevalue->offset   = scalar_uint64;
            return val;
        }
        case MilValueBuilder::Type::ARRAY_SHAPE:
        case MilValueBuilder::Type::ARRAY_INT32: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32,
                                                           {(int64_t)array_int32.size()});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_INTS;
            val->immediatevalue->tensor->ints =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedInts>();
            val->immediatevalue->tensor->ints->n_values = array_int32.size();
            val->immediatevalue->tensor->ints->values   = (int32_t*)malloc(array_int32.size() * sizeof(int32_t));
            std::copy(array_int32.begin(), array_int32.end(), val->immediatevalue->tensor->ints->values);
            return val;
        }
        case MilValueBuilder::Type::ARRAY_FLOAT: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16,
                                                           {(int64_t)array_float.size()});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_BYTES;
            val->immediatevalue->tensor->bytes =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedBytes>();
            auto size                                       = array_float.size() * sizeof(int16_t);
            val->immediatevalue->tensor->bytes->values.len  = size;
            val->immediatevalue->tensor->bytes->values.data = (uint8_t*)malloc(size);
            int16_t* dstPtr                                 = (int16_t*)val->immediatevalue->tensor->bytes->values.data;
            MNNQuantizeFP16((const float*)array_float.data(), dstPtr, array_float.size());
            return val;
        }
        case MilValueBuilder::Type::ARRAY_BOOL: {
            val->type                       = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL,
                                                                       {(int64_t)array_bool.size()});
            val->value_case                 = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue             = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_BOOLS;
            val->immediatevalue->tensor->bools =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedBools>();
            val->immediatevalue->tensor->bools->n_values = array_bool.size();
            val->immediatevalue->tensor->bools->values   = (int*)malloc(array_bool.size() * sizeof(int32_t));
            for (int i = 0; i < array_bool.size(); i++) {
                val->immediatevalue->tensor->bools->values[i] = array_bool[i];
            }
            return val;
        }
        case MilValueBuilder::Type::SCALAR_INT32: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32, {});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_INTS;
            val->immediatevalue->tensor->ints =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedInts>();
            val->immediatevalue->tensor->ints->n_values  = 1;
            val->immediatevalue->tensor->ints->values    = (int32_t*)malloc(sizeof(int32_t));
            val->immediatevalue->tensor->ints->values[0] = scalar_int32;
            return val;
        }
        case MilValueBuilder::Type::SCALAR_FLOAT32: {
            val->type = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, {});
            val->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            auto tensor = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor = tensor;
            tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_FLOATS;
            tensor->floats = create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedFloats>();
            tensor->floats->n_values = 1;
            tensor->floats->values = create<float>(1);
            tensor->floats->values[0] = scalar_float;
            return val;
        }
        case MilValueBuilder::Type::SCALAR_FLOAT: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16, {});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_BYTES;
            val->immediatevalue->tensor->bytes =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedBytes>();
            val->immediatevalue->tensor->bytes->values.len  = sizeof(int16_t);
            val->immediatevalue->tensor->bytes->values.data = (uint8_t*)malloc(sizeof(int16_t));
            int16_t* dstPtr                                 = (int16_t*)val->immediatevalue->tensor->bytes->values.data;
            MNNQuantizeFP16(&scalar_float, dstPtr, 1);
            return val;
        }
        case MilValueBuilder::Type::SCALAR_STRING: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__STRING, {});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_STRINGS;
            val->immediatevalue->tensor->strings =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedStrings>();
            val->immediatevalue->tensor->strings->n_values  = 1;
            val->immediatevalue->tensor->strings->values    = (char**)malloc(sizeof(char*));
            val->immediatevalue->tensor->strings->values[0] = s_strdup(scalar_string);
            return val;
        }
        case MilValueBuilder::Type::SCALAR_BOOL: {
            val->type           = createMilValueTypeTensor(CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL, {});
            val->value_case     = CORE_ML__SPECIFICATION__MILSPEC__VALUE__VALUE_IMMEDIATE_VALUE;
            val->immediatevalue = create_and_init<CoreML__Specification__MILSpec__Value__ImmediateValue>();
            val->immediatevalue->value_case = CORE_ML__SPECIFICATION__MILSPEC__VALUE__IMMEDIATE_VALUE__VALUE_TENSOR;
            val->immediatevalue->tensor     = create_and_init<CoreML__Specification__MILSpec__TensorValue>();
            val->immediatevalue->tensor->value_case = CORE_ML__SPECIFICATION__MILSPEC__TENSOR_VALUE__VALUE_BOOLS;
            val->immediatevalue->tensor->bools =
                create_and_init<CoreML__Specification__MILSpec__TensorValue__RepeatedBools>();
            val->immediatevalue->tensor->bools->n_values  = 1;
            val->immediatevalue->tensor->bools->values    = (int*)malloc(sizeof(int));
            val->immediatevalue->tensor->bools->values[0] = scalar_bool;
            return val;
        }
        default:
            break;
    }
    return nullptr;
}

// --- MilOperationBuilder ---
MilOperationBuilder::MilOperationBuilder() {
}
MilOperationBuilder::MilOperationBuilder(const std::string& type) {
    mType = type;
}
MilOperationBuilder::~MilOperationBuilder() {
}
void MilOperationBuilder::setType(const std::string& type) {
    mType = type;
}
void MilOperationBuilder::setName(const std::string& name) {
    mName = name;
}
std::string MilOperationBuilder::getType() {
    return mType;
}
std::string MilOperationBuilder::getName() {
    return mName;
}
void MilOperationBuilder::addInput(const std::string& param_name, const std::string& tensor_name) {
    mInputs[param_name] = {tensor_name};
}
void MilOperationBuilder::addInputs(const std::string& param_name, const std::vector<std::string>& tensor_names) {
    mInputs[param_name] = tensor_names;
}
void MilOperationBuilder::addOutput(const std::string& name, CoreML__Specification__MILSpec__DataType data_type,
                                    const std::vector<int64_t>& shape) {
    std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
    info->name      = name;
    info->data_type = data_type;
    info->shape     = shape;
    info->type      = MilTensorInfo::Type::TENSOR;
    mOutputs.emplace_back(std::move(info));
}
void MilOperationBuilder::addAttribute(const std::string& attr_name, std::unique_ptr<MilValueBuilder> attr_value) {
    mAttributes[attr_name] = std::move(attr_value);
}
void MilOperationBuilder::update(MilOperationBuilder* other) {
    for (size_t i = 0; i < mOutputs.size(); i++) {
        updateDynamicShape(mOutputs[i]->shape, other->mOutputs[i]->shape);
    }
    for (auto& pair : mAttributes) {
        auto& attr_name  = pair.first;
        auto& attr_value = pair.second;
        if (attr_value->type != MilValueBuilder::Type::ARRAY_SHAPE) {
            continue;
        }
        if (other->mAttributes.find(attr_name) == other->mAttributes.end()) {
            continue;
        }
        const auto& other_attr_value = other->mAttributes[attr_name];
        updateDynamicShape(attr_value->array_int32, other_attr_value->array_int32);
    }
}
CoreML__Specification__MILSpec__Operation* MilOperationBuilder::build() {
    auto op  = create_and_init<CoreML__Specification__MILSpec__Operation>();
    op->type = s_strdup(mType);
    // build inputs
    op->n_inputs = mInputs.size();
    if (op->n_inputs > 0) {
        op->inputs = (CoreML__Specification__MILSpec__Operation__InputsEntry**)malloc(
            sizeof(CoreML__Specification__MILSpec__Operation__InputsEntry*) * op->n_inputs);
        size_t i = 0;
        for (const auto& pair : mInputs) {
            auto entry                = create_and_init<CoreML__Specification__MILSpec__Operation__InputsEntry>();
            entry->key                = s_strdup(pair.first);
            auto argNames             = pair.second;
            entry->value              = create_and_init<CoreML__Specification__MILSpec__Argument>();
            entry->value->n_arguments = argNames.size();
            if (!argNames.empty()) {
                entry->value->arguments = create<CoreML__Specification__MILSpec__Argument__Binding*>(argNames.size());
                for (auto i = 0; i < argNames.size(); ++i) {
                    entry->value->arguments[i] = create_and_init<CoreML__Specification__MILSpec__Argument__Binding>();
                    entry->value->arguments[i]->binding_case =
                        CORE_ML__SPECIFICATION__MILSPEC__ARGUMENT__BINDING__BINDING_NAME;
                    entry->value->arguments[i]->name = s_strdup(argNames[i]);
                }
            }
            op->inputs[i++] = entry;
        }
    }
    // build outputs
    op->n_outputs = mOutputs.size();
    if (op->n_outputs > 0) {
        op->outputs = (CoreML__Specification__MILSpec__NamedValueType**)malloc(
            sizeof(CoreML__Specification__MILSpec__NamedValueType*) * op->n_outputs);
        for (size_t i = 0; i < mOutputs.size(); ++i) {
            const auto& name = mOutputs[i]->name;
            auto data_type   = mOutputs[i]->data_type;
            auto shape       = mOutputs[i]->shape;
            op->outputs[i]   = createMilNamedValueType(name, createMilValueTypeTensor(data_type, shape));
        }
    }
    // build attributes
    op->n_attributes = mAttributes.size();
    if (op->n_attributes > 0) {
        op->attributes = (CoreML__Specification__MILSpec__Operation__AttributesEntry**)malloc(
            sizeof(CoreML__Specification__MILSpec__Operation__AttributesEntry*) * op->n_attributes);
        size_t i = 0;
        for (const auto& pair : mAttributes) {
            auto entry          = create_and_init<CoreML__Specification__MILSpec__Operation__AttributesEntry>();
            entry->key          = s_strdup(pair.first);
            entry->value        = pair.second->build();
            op->attributes[i++] = entry;
        }
    }
    return op;
}

MlModelBuilder::MlModelBuilder(int32_t version) : mVersion(version) {
}
MlModelBuilder::~MlModelBuilder() {
    if (mModel) {
        core_ml__specification__model__free_unpacked(mModel, nullptr);
    }
}

void MlModelBuilder::addInput(std::unique_ptr<MilTensorInfo> info) {
    auto name = info->name;
    auto iter = mIOMap.find(name);
    if (iter != mIOMap.end()) {
        iter->second->update(info.get());
        return;
    }
    mInputs.push_back(std::move(info));
    mIOMap.insert({name, mInputs.back().get()});
}

void MlModelBuilder::addOutput(std::unique_ptr<MilTensorInfo> info) {
    auto name = info->name;
    auto iter = mIOMap.find(name);
    if (iter != mIOMap.end()) {
        iter->second->update(info.get());
        return;
    }
    mOutputs.push_back(std::move(info));
    mIOMap.insert({name, mOutputs.back().get()});
}

void MlModelBuilder::addOperation(std::unique_ptr<MilOperationBuilder> builer) {
    auto name = builer->getName();
    auto iter = mOperationBuilderMap.find(name);
    if (iter != mOperationBuilderMap.end()) {
        iter->second->update(builer.get());
        return;
    }
    mOperationBuilders.push_back(std::move(builer));
    mOperationBuilderMap.insert({name, mOperationBuilders.back().get()});
}

std::unique_ptr<MilOperationBuilder> MilOperationBuilder::clone(const std::map<std::string, std::string>& names,
                                                                const std::string& prefix) const {
    auto rename = [&](const std::string& name) {
        auto it = names.find(name);
        return it == names.end() ? prefix + name : it->second;
    };
    std::unique_ptr<MilOperationBuilder> result(new MilOperationBuilder(mType));
    result->mName = prefix + mName;
    for (const auto& input : mInputs) {
        for (const auto& name : input.second) {
            result->mInputs[input.first].push_back(rename(name));
        }
    }
    for (const auto& output : mOutputs) {
        std::unique_ptr<MilTensorInfo> info(new MilTensorInfo(*output));
        info->name = rename(info->name);
        result->mOutputs.push_back(std::move(info));
    }
    for (const auto& attr : mAttributes) {
        result->mAttributes[attr.first].reset(new MilValueBuilder(*attr.second));
    }
    return result;
}
bool MilOperationBuilder::usesInput(const std::string& name) const {
    for (const auto& input : mInputs) {
        for (const auto& value : input.second) {
            if (value == name) {
                return true;
            }
        }
    }
    return false;
}
bool MlModelBuilder::usesInput(const std::string& name) const {
    for (const auto& operation : mOperationBuilders) {
        if (operation->usesInput(name)) {
            return true;
        }
    }
    return false;
}
void MlModelBuilder::appendOperations(const MlModelBuilder& source, const std::map<std::string, std::string>& names,
                                      const std::string& prefix) {
    for (const auto& operation : source.mOperationBuilders) {
        addOperation(operation->clone(names, prefix));
    }
}

void MlModelBuilder::markStaticOperations() {
    mStaticOperationCount = mOperationBuilders.size();
}

void MlModelBuilder::resetOperations() {
    mInputs.clear();
    mOutputs.clear();
    mIOMap.clear();
    mOperationBuilders.resize(mStaticOperationCount);
    mOperationBuilderMap.clear();
    for (auto& op : mOperationBuilders) {
        mOperationBuilderMap.emplace(op->getName(), op.get());
    }
}

CoreML__Specification__MILSpec__Block* MlModelBuilder::buildBlock() {
    auto block = create_and_init<CoreML__Specification__MILSpec__Block>();
    // outputs
    block->n_outputs = mOutputs.size();
    if (block->n_outputs > 0) {
        block->outputs = (char**)malloc(sizeof(char*) * block->n_outputs);
        for (size_t i = 0; i < block->n_outputs; ++i) {
            block->outputs[i] = s_strdup(mOutputs[i]->name);
        }
    }
    // Public multiarrays have rank >= 1; restore scalar values inside the MIL function.
    std::map<std::string, std::string> scalars;
    std::vector<CoreML__Specification__MILSpec__Operation*> operations;
    for (const auto& input : mInputs) {
        if (input->type != MilTensorInfo::Type::INPUT || !input->shape.empty()) {
            continue;
        }
        auto name = input->name + "_scalar";
        auto axesName = name + "_axes";
        scalars[input->name] = name;
        MilOperationBuilder axes("const");
        axes.setName(axesName);
        axes.addOutput(axesName, CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32, {1});
        axes.addAttribute("val", MilValueBuilder::createMilValueBuilder(std::vector<int>{0}));
        operations.push_back(axes.build());
        MilOperationBuilder squeeze("squeeze");
        squeeze.setName(name);
        squeeze.addInput("x", input->name);
        squeeze.addInput("axes", axesName);
        squeeze.addOutput(name, input->data_type, {});
        operations.push_back(squeeze.build());
    }
    for (const auto& op : mOperationBuilders) {
        operations.push_back(scalars.empty() ? op->build() : op->clone(scalars, "")->build());
    }
    block->n_operations = operations.size();
    block->operations = create<CoreML__Specification__MILSpec__Operation*>(operations.size());
    std::copy(operations.begin(), operations.end(), block->operations);
    return block;
}

CoreML__Specification__MILSpec__Function* MlModelBuilder::buildFunction() {
    auto function   = create_and_init<CoreML__Specification__MILSpec__Function>();
    function->opset = s_strdup("CoreML8");
    // inputs
    function->n_inputs = mInputs.size();
    if (function->n_inputs > 0) {
        function->inputs = (CoreML__Specification__MILSpec__NamedValueType**)malloc(
            sizeof(CoreML__Specification__MILSpec__NamedValueType*) * function->n_inputs);
        for (size_t i = 0; i < function->n_inputs; ++i) {
            auto name  = mInputs[i]->name;
            auto dtype = mInputs[i]->data_type;
            auto shape = mInputs[i]->shape;
            if (shape.empty() && mInputs[i]->type == MilTensorInfo::Type::INPUT) {
                shape = {1};
            }
            if (mInputs[i]->type == MilTensorInfo::Type::STATE) {
                function->inputs[i] = createMilNamedValueType(name, createMilValueTypeState(dtype, shape));
            } else {
                function->inputs[i] = createMilNamedValueType(name, createMilValueTypeTensor(dtype, shape));
            }
        }
    }
    // block
    function->n_block_specializations = 1;
    function->block_specializations   = (CoreML__Specification__MILSpec__Function__BlockSpecializationsEntry**)malloc(
        sizeof(CoreML__Specification__MILSpec__Function__BlockSpecializationsEntry*) *
        function->n_block_specializations);
    auto entry   = create_and_init<CoreML__Specification__MILSpec__Function__BlockSpecializationsEntry>();
    entry->key   = s_strdup("CoreML8");
    entry->value = buildBlock();
    function->block_specializations[0] = entry;
    return function;
}

CoreML__Specification__MILSpec__Program* MlModelBuilder::buildProgram() {
    auto program         = create_and_init<CoreML__Specification__MILSpec__Program>();
    program->version     = 1;
    program->n_functions = 1;
    program->functions   = (CoreML__Specification__MILSpec__Program__FunctionsEntry**)malloc(
        sizeof(CoreML__Specification__MILSpec__Program__FunctionsEntry*) * program->n_functions);
    auto entry            = create_and_init<CoreML__Specification__MILSpec__Program__FunctionsEntry>();
    entry->key            = s_strdup("main");
    entry->value          = buildFunction();
    program->functions[0] = entry;
    return program;
}

CoreML__Specification__FeatureDescription* MlModelBuilder::buildFeatureDescription(const MilTensorInfo* info) {
    auto desc                                      = create_and_init<CoreML__Specification__FeatureDescription>();
    desc->name                                     = s_strdup(info->name);
    desc->type                                     = create_and_init<CoreML__Specification__FeatureType>();
    CoreML__Specification__ArrayFeatureType* array = nullptr;
    if (info->type == MilTensorInfo::Type::STATE) {
        desc->type->type_case            = CORE_ML__SPECIFICATION__FEATURE_TYPE__TYPE_STATE_TYPE;
        desc->type->statetype            = create_and_init<CoreML__Specification__StateFeatureType>();
        desc->type->statetype->type_case = CORE_ML__SPECIFICATION__STATE_FEATURE_TYPE__TYPE_ARRAY_TYPE;
        desc->type->statetype->arraytype = create_and_init<CoreML__Specification__ArrayFeatureType>();
        array                            = desc->type->statetype->arraytype;
    } else {
        desc->type->type_case      = CORE_ML__SPECIFICATION__FEATURE_TYPE__TYPE_MULTI_ARRAY_TYPE;
        desc->type->multiarraytype = create_and_init<CoreML__Specification__ArrayFeatureType>();
        array                      = desc->type->multiarraytype;
    }
    if (info->data_type == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32) {
        array->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__INT32;
    } else {
        array->datatype = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__ARRAY_DATA_TYPE__FLOAT16;
    }
    if (info->type == MilTensorInfo::Type::OUTPUT) {
        return desc;
    }
    if (info->shape.empty() && info->type == MilTensorInfo::Type::INPUT) {
        array->n_shape = 1;
        array->shape = create<int64_t>(1);
        array->shape[0] = 1;
        return desc;
    }
    bool isSizeRange = false;
    array->n_shape   = info->shape.size();
    array->shape     = create<int64_t>(array->n_shape);
    for (int i = 0; i < info->shape.size(); i++) {
        if (info->shape[i] < 0) {
            array->shape[i] = 1;
            isSizeRange     = true;
        } else {
            array->shape[i] = info->shape[i];
        }
    }
    if (isSizeRange) {
        array->shape_flexibility_case   = CORE_ML__SPECIFICATION__ARRAY_FEATURE_TYPE__SHAPE_FLEXIBILITY_SHAPE_RANGE;
        array->shaperange               = create_and_init<CoreML__Specification__ArrayFeatureType__ShapeRange>();
        array->shaperange->n_sizeranges = info->shape.size();
        array->shaperange->sizeranges   = create<CoreML__Specification__SizeRange*>(info->shape.size());
        for (int i = 0; i < info->shape.size(); i++) {
            array->shaperange->sizeranges[i] = create_and_init<CoreML__Specification__SizeRange>();
            if (info->shape[i] < 0) {
                array->shaperange->sizeranges[i]->lowerbound = 1;
                array->shaperange->sizeranges[i]->upperbound = 4096;
            } else {
                array->shaperange->sizeranges[i]->lowerbound = info->shape[i];
                array->shaperange->sizeranges[i]->upperbound = info->shape[i];
            }
        }
    }
    return desc;
}

CoreML__Specification__Model* MlModelBuilder::build() {
    if (mModel) {
        core_ml__specification__model__free_unpacked(mModel, nullptr);
    }
    mModel                       = create_and_init<CoreML__Specification__Model>();
    mModel->specificationversion = mVersion;
    mModel->type_case            = CORE_ML__SPECIFICATION__MODEL__TYPE_ML_PROGRAM;
    mModel->mlprogram            = buildProgram();
    mModel->description          = create_and_init<CoreML__Specification__ModelDescription>();
    // inputs and states
    for (const auto& input : mInputs) {
        if (input->type == MilTensorInfo::Type::STATE) {
            ++mModel->description->n_state;
        } else {
            ++mModel->description->n_input;
        }
    }
    mModel->description->input = create<CoreML__Specification__FeatureDescription*>(mModel->description->n_input);
    mModel->description->state = create<CoreML__Specification__FeatureDescription*>(mModel->description->n_state);
    size_t inputIndex = 0, stateIndex = 0;
    for (const auto& input : mInputs) {
        auto desc = buildFeatureDescription(input.get());
        if (input->type == MilTensorInfo::Type::STATE) {
            mModel->description->state[stateIndex++] = desc;
        } else {
            mModel->description->input[inputIndex++] = desc;
        }
    }
    // model output
    mModel->description->n_output = mOutputs.size();
    mModel->description->output   = create<CoreML__Specification__FeatureDescription*>(mOutputs.size());
    for (size_t i = 0; i < mOutputs.size(); ++i) {
        mModel->description->output[i] = buildFeatureDescription(mOutputs[i].get());
    }
    return mModel;
}

constexpr uint64_t DefaultStorageAlignment = 64;
constexpr uint64_t BlobMetadataSentinel    = 0xDEADBEEF;
struct blob_metadata {
    uint32_t sentinel             = BlobMetadataSentinel;  // for validating correctness of metadata.
    BlobDataType mil_dtype        = BlobDataType::Float16; // data type of the blob data.
    uint64_t sizeInBytes          = 0;                     // size of the blob data in bytes.
    uint64_t offset               = 0;                     // offset in file for blob data.
    uint64_t padding_size_in_bits = 0;                     // describes the number of unused bits in this blob,
                                                           // required to calculate the actual size for spans of
                                                           // sub-btye-sized types.  Unused otherwise
    // Reserve fields
    uint64_t reserved_1 = 0;
    uint64_t reserved_2 = 0;
    uint64_t reserved_3 = 0;
    uint64_t reserved_4 = 0;
};

MilWeightWriter::MilWeightWriter(const std::string& filePath) : mFilePath(filePath) {
    mFile = fopen(mFilePath.c_str(), "wb");
    mValid = mFile != nullptr;
    flushHeader();
}

MilWeightWriter::MilWeightWriter(const std::shared_ptr<std::vector<uint8_t>>& data, size_t capacity) : mData(data) {
    mValid = mData != nullptr;
    if (mValid) {
        mData->reserve(capacity);
        mData->resize(sizeof(mHeader), 0);
        flushHeader();
    }
}

MilWeightWriter::~MilWeightWriter() {
    if (mFile) {
        fclose(mFile);
        mFile = nullptr;
    }
}

uint64_t MilWeightWriter::tell() const {
    if (mData) {
        return mData->size();
    }
    if (!mFile) {
        return 0;
    }
    return static_cast<uint64_t>(std::ftell(mFile));
}

void MilWeightWriter::writeRaw(const void* p, size_t sz) {
    if (mData) {
        const auto* data = static_cast<const uint8_t*>(p);
        mData->insert(mData->end(), data, data + sz);
        return;
    }
    if (mValid && std::fwrite(p, 1, sz, mFile) != sz) {
        mValid = false;
    }
}

void MilWeightWriter::flushHeader() {
    if (mData) {
        ::memcpy(mData->data(), &mHeader, sizeof(mHeader));
        return;
    }
    if (!mValid || std::fseek(mFile, 0, SEEK_SET) != 0) {
        mValid = false;
        return;
    }
    writeRaw(&mHeader, sizeof(mHeader));
    if (std::fflush(mFile) != 0 || std::fseek(mFile, 0, SEEK_END) != 0) {
        mValid = false;
    }
}

void MilWeightWriter::alignFile() {
    if (!mValid) {
        return;
    }
    uint64_t pos = tell();
    uint64_t pad = (DefaultStorageAlignment - (pos % DefaultStorageAlignment)) % DefaultStorageAlignment;
    if (pad) {
        char zero[DefaultStorageAlignment] = {0};
        writeRaw(zero, pad);
    }
}

BlobDataType toBlobDataType(CoreML__Specification__MILSpec__DataType data_type) {
    switch (data_type) {
        // FLOAT
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16:
            return BlobDataType::Float16;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32:
            return BlobDataType::Float32;
        // INT
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT4:
            return BlobDataType::Int4;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT8:
            return BlobDataType::Int8;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT16:
            return BlobDataType::Int16;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32:
            return BlobDataType::Int32;
        // UINT
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT1:
            return BlobDataType::UInt1;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT3:
            return BlobDataType::UInt3;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT4:
            return BlobDataType::UInt4;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT6:
            return BlobDataType::UInt6;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT8:
            return BlobDataType::UInt8;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT16:
            return BlobDataType::UInt16;
        case CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT32:
            return BlobDataType::UInt32;
        default:
            return BlobDataType::Float16;
    }
}

uint64_t MilWeightWriter::write(const void* data, size_t size, CoreML__Specification__MILSpec__DataType dtype) {
    if (!mValid) {
        return 0;
    }
    uint64_t meta_offset = tell();
    blob_metadata metadata;
    metadata.mil_dtype            = toBlobDataType(dtype);
    metadata.sizeInBytes          = size;
    metadata.offset               = meta_offset + sizeof(metadata);
    metadata.padding_size_in_bits = 0;
    writeRaw(&metadata, sizeof(metadata));
    writeRaw(data, size);
    alignFile();
    // blob count is header
    mHeader.count++;
    flushHeader();
    return meta_offset;
}


} // namespace MNN
