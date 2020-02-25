#include "backend/cpu/CPUBackend.hpp"
namespace MNN {
    class CPUGatherND : public Execution {
    public:
        CPUGatherND(Backend* bn) : Execution(bn) {
            //Do nothing;
        }
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    private:
        std::vector<int> mDimsToCount;
        int mSliceN = 0;
        int mSliceSize = 0;
    };
}
