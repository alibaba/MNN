//
//  Optimizer.hpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Optimizer_hpp
#define Optimizer_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/MNNForwardType.h>

namespace MNN {
namespace Express {
class MNN_PUBLIC Optimizer {
public:
    enum Device {
        CPU = 0,
        GPU = 1,
        OTHER = 2,
        AUTO = 3
    };
    struct MNN_PUBLIC Config {
        Device device = CPU;
        MNNForwardType forwardType = MNN_FORWARD_ALL;
        int numThread = 4;
    };
    static std::shared_ptr<Optimizer> create(Config config);
    struct MNN_PUBLIC Cost {
        float compute; // MFlops
        float memory;  // MB
    };
    class MNN_PUBLIC Parameters {
    public:
        Parameters(int n);
        virtual ~Parameters();

        float* get() const;
        int size() const;

    private:
        float* mValue;
        int mSize;
    };
    virtual std::shared_ptr<Parameters> onGetParameters(const std::vector<VARP>& outputs);

    //Given paramters and measure cost, the parameters must be the same as onGetParameters
    virtual Cost onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) = 0;

    //Modify the output directly, the parameters must be the same as onGetParameters
    virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) = 0;

    Optimizer() = default;
    virtual ~Optimizer() = default;
};
} // namespace Express
} // namespace MNN
#endif
