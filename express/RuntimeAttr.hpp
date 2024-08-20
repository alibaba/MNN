#ifndef RuntimeAttr_hpp
#define RuntimeAttr_hpp
#include "core/Session.hpp"
namespace MNN{
namespace Express {
struct Cache {
    AutoStorage<uint8_t> modelBuffer;
    AutoStorage<uint8_t> cacheBuffer;
    size_t cacheOffset = 0;
    std::string cacheFile;
    size_t lastCacheSize = 0;
};
struct RuntimeAttr {
    Session::ModeGroup modes;
    RuntimeInfo mRuntime;
    std::shared_ptr<Runtime> mInfo;
    std::shared_ptr<Cache> mCache;
    BackendConfig mConfig;
    bool mUserConfig;
    int mNumberThread;
    // Use for static module to compute flops
    float mFlops;
    std::string mExternalFile;
};
struct ExecutorAttr {
    std::shared_ptr<Backend> constantBackend;
    MNNForwardType firstType;
    std::string externalFile;
};
};
};


#endif
