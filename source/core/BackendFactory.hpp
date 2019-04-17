//
//  BackendFactory.hpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BackendFactory_hpp
#define BackendFactory_hpp

#include "Backend.hpp"

namespace MNN {
/** backend factory */
class BackendFactory {
public:
    /**
     * @brief create backend with given info.
     * @param info backend info.
     * @return created backend or NULL if failed.
     */
    static Backend* create(const Backend::Info& info);
};
} // namespace MNN

#endif /* BackendFactory_hpp */
