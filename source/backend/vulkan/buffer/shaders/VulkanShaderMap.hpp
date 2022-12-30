//
//  VulkanShaderMap.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanShaderMap_hpp
#define VulkanShaderMap_hpp

#include <map>
#include <string>
namespace MNN {
class VulkanShaderMap {
public:
    VulkanShaderMap() {
        init();
    }
    ~VulkanShaderMap() {
    }

    void init();

    std::pair<const unsigned char*, size_t> search(const std::string& key) {
        auto iter = mMaps.find(key);
        if (iter != mMaps.end()) {
            return iter->second;
        }
        return std::make_pair(nullptr, 0);
    }

private:
    std::map<std::string, std::pair<const unsigned char*, size_t>> mMaps;
};
} /*namespace MNN*/
#endif /* VulkanShaderMap_hpp */
