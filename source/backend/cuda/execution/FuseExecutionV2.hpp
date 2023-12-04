#ifndef FuseExecutionV2_hpp
#define FuseExecutionV2_hpp
namespace MNN {
namespace CUDA {
class FuseExecutionV2 {
public:
    static bool check(const Op* op);

    static Execution* create(const Op* op, Backend *backend, int inputSize, int outputSize);

};
}
};

#endif
