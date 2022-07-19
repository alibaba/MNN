//
//  ShaderMap.hpp
//  MNN
//
//  Created by MNN on 2022/05/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ShaderMap_hpp
#define ShaderMap_hpp

#include <map>
#include <string>
namespace MNN {
class ShaderMap {
public:
    ShaderMap() {
        init();
    }
    ~ShaderMap() {
    }

    void init();

    const char* search(const std::string& key) {
        auto iter = mMaps.find(key);
        if (iter != mMaps.end()) {
            return iter->second;
        }
        return nullptr;
    }
    const std::map<std::string, const char*>& getRaw() const {
        return mMaps;
    }

private:
    std::map<std::string, const char*> mMaps;
};
} /*namespace MNN*/
#endif /* ShaderMap_hpp */
