#ifndef ConvertExecution_hpp
#define ConvertExecution_hpp
#include "core/Execution.hpp"
namespace MNN {
class ConvertExecution : public Execution {
public:
    ConvertExecution(Backend* bn, const Op* op);
    virtual ~ConvertExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }
private:
    const Op* mOp;
    uint8_t* mHostPtr = nullptr;
};
};
#endif
