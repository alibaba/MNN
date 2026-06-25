#ifndef RKNN_BUNDLE_HPP
#define RKNN_BUNDLE_HPP

#include <memory>
#include <string>

#include "config.hpp"

namespace MNN {
struct NetT;

struct RKNNBundlePaths {
    std::string rknnPath;
    std::string manifestPath;
};

bool PopulateRKNNConfigFromEnv(modelConfig& modelPath);
bool GenerateRKNNBundle(const modelConfig& modelPath, RKNNBundlePaths* bundlePaths);
std::unique_ptr<NetT> BuildRKNNWrapperNet(const NetT& sourceNet, const modelConfig& modelPath,
                                          const RKNNBundlePaths& bundlePaths);
}

#endif
