#ifndef ConvertTflite_hpp
#define ConvertTflite_hpp
#include <memory>
#include <map>
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
#include "../../../tools/converter/source/tflite/schema/schema_generated.h"
namespace MNN {

class NeuropilotBackend;
class ConvertTflite {
public:
    ConvertTflite();
    ~ ConvertTflite();
    struct Command {
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        std::unique_ptr<tflite::OperatorT> op;
    };
    struct CommandBuffer {
        std::vector<Command> commands;
        const Op* op;
        std::vector<std::shared_ptr<Tensor>> extraConst;
    };
    CommandBuffer convert(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    class Convert {
    public:
        Convert() = default;
        virtual ~Convert() = default;
        virtual CommandBuffer onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) = 0;
    };
    static tflite::TensorType getType(const Tensor* tensor);
    static std::vector<int> getShapeOfTensor(const Tensor* tensor);
    Tensor* makeReshape(CommandBuffer& res, Tensor* tensor, std::vector<int> shape, Tensor* outputUser = nullptr);
    Tensor* makeTranspose(CommandBuffer& res, Tensor* tensor, std::vector<int> dims);
    Tensor* makeTile(CommandBuffer& res, Tensor* tensor, std::vector<int> dims);
    Tensor* makeBinary(CommandBuffer& res, Tensor* A, Tensor* B, tflite::BuiltinOperator operation);
    Tensor* makeSoftmax(CommandBuffer& res, Tensor* tensor);
    Tensor* makeConcat(CommandBuffer& res, std::vector<Tensor*> inputs, int axis);
    void makeMatMul(CommandBuffer& res, Tensor* A, Tensor* B, bool adjA, bool adB, Tensor* dst);
    Tensor* makeSlice(CommandBuffer& res, Tensor* input, int sta, int size, int axis);

    std::vector<std::unique_ptr<tflite::OperatorCodeT>> releaseCodes() {
        mOperatorCodeIndexMap.clear();
        return std::move(mOperatorCodes);
    }
    int getOpIndex(tflite::BuiltinOperator op);
    int getCustomOpIndex(std::string name);
    static std::shared_ptr<Tensor> getIntArrayTensor(std::vector<int> shapes);
    NeuropilotBackend* pBackend = nullptr;
private:
    std::map<tflite::BuiltinOperator, int> mOperatorCodeIndexMap;
    std::map<std::string, int> mCustomOpIndex;
    std::vector<std::unique_ptr<tflite::OperatorCodeT>> mOperatorCodes;
    std::map<OpType, std::shared_ptr<Convert>> mConverters;
};
}

#endif
