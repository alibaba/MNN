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
    struct Immutable {
        Session::ModeGroup modes;
        BackendConfig mConfig;
        bool mUserConfig;
        int mNumberThread;
        std::string mExternalFile;
    };
    std::shared_ptr<Immutable> mContent;
    RuntimeInfo mRuntime;
    std::shared_ptr<Runtime> mInfo;
    std::shared_ptr<Cache> mCache;
    // Use for static module to compute flops
    float mFlops;
    mutable int mResizeStatus = 0;
};
struct ExecutorAttr {
    std::shared_ptr<Backend> constantBackend;
    MNNForwardType firstType;
    int numThread = 1;
    BackendConfig config;
    std::string externalFile;
};
};
};


#endif
