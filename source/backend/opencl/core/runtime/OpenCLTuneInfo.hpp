#ifndef OpenCLTuneInfo_hpp
#define OpenCLTuneInfo_hpp
#include "CLCache_generated.h"
namespace MNN {
namespace OpenCL {
struct TuneInfo {
    std::vector<std::unique_ptr<CLCache::OpInfoT>> mInfos;
};
}
}

#endif
