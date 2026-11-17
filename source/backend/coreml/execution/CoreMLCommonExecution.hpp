//
//  CoreMLCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLCOMMONEXECUTION_HPP
#define MNN_COREMLCOMMONEXECUTION_HPP
#include <cstdint>
#include <memory>
#include <type_traits>
#include "CoreMLBackend.hpp"
#include "core/Execution.hpp"

namespace MNN {

class CoreMLCommonExecution : public Execution {
public:
    CoreMLCommonExecution(Backend* backend, const Op* op);
    virtual ~CoreMLCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

protected:
    // Own a single node until output() or finish() explicitly appends it to the graph.
    class Operation {
    public:
        Operation(CoreMLCommonExecution& execution, const std::string& type);
        Operation(Operation&&) = default;
        Operation(const Operation&) = delete;
        Operation& operator=(const Operation&) = delete;
        Operation& input(const std::string& parameter, const Tensor* tensor);
        Operation& input(const std::string& parameter, const std::string& value);
        Operation& inputs(const std::string& parameter, const std::vector<std::string>& values);
        Operation& inputs(const std::string& parameter, const std::vector<Tensor*>& tensors);
        template <typename T>
        Operation& scalar(const std::string& parameter, T value, const std::string& name = "") {
            static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value ||
                              std::is_same<T, bool>::value || std::is_same<T, std::string>::value,
                          "Unsupported MIL scalar type");
            return input(parameter, mExecution.buildScalar(value, name.empty() ? parameter : name));
        }
        Operation& scalar(const std::string& parameter, const char* value, const std::string& name = "") {
            return scalar(parameter, std::string(value), name);
        }
        template <typename T>
        Operation& array(const std::string& parameter, const std::vector<T>& values, const std::string& name = "") {
            return input(parameter, mExecution.buildArray(values, name.empty() ? parameter : name));
        }
        Operation& shape(const std::string& parameter, const std::vector<int>& values, const std::string& name = "");
        Operation& scalarFloat32(const std::string& parameter, float value, const std::string& name = "");
        void output(const std::vector<Tensor*>& outputs);
        std::string
        output(const std::vector<int>& shape, const std::string& name,
               CoreML__Specification__MILSpec__DataType dtype = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16);
        // State writes have no result tensor, but still require a stable node name.
        void finish(const std::string& name);

    private:
        CoreMLCommonExecution& mExecution;
        std::unique_ptr<MilOperationBuilder> mBuilder;
    };
    Operation operation(const std::string& type) { return Operation(*this, type); }
    std::string buildBlob(const void* data, size_t size, CoreML__Specification__MILSpec__DataType data_type,
                          const std::vector<int>& shape, std::string name) {
        return mCoreMLBackend->buildBlobConst(data, size, data_type, shape, mName + "_" + name);
    }
    template <typename T>
    std::string buildArray(const std::vector<T>& data, std::string name) {
        return mCoreMLBackend->buildArray<T>(data, mName + "_" + name);
    }
    template <typename T>
    std::string buildScalar(const T data, std::string name) {
        return mCoreMLBackend->buildScalar<T>(data, mName + "_" + name);
    }
    std::string buildShape(const std::vector<int>& data, std::string name) {
        return mCoreMLBackend->buildShape(data, mName + "_" + name);
    }
    CoreMLBackend* mCoreMLBackend;
    const Op* mOp;
    std::string mName;

private:
    void addOperation(std::unique_ptr<MilOperationBuilder> builder, const std::vector<Tensor*>& outputs);
    std::string addOperation(std::unique_ptr<MilOperationBuilder> builder, const std::vector<int>& shape,
                             const std::string& name, CoreML__Specification__MILSpec__DataType dtype);
};

} // namespace MNN
#endif // MNN_COREMLCOMMONEXECUTION_HPP
