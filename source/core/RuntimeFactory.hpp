//
//  RuntimeFactory.hpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RuntimeFactory_hpp
#define RuntimeFactory_hpp

#include "core/Backend.hpp"

namespace MNN {
/** Runtime factory */
class RuntimeFactory {
public:
    /**
     * @brief create backend with given info.
     * @param info backend info.
     * @return created backend or NULL if failed.
     */
    static Runtime* create(const Backend::Info& info);
};
} // namespace MNN

#endif /* RuntimeFactory_hpp */
