//
//  NonCopyable.hpp
//  MNN
//
//  Created by MNN on 2018/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NonCopyable_hpp
#define NonCopyable_hpp

namespace MNN {
/** protocol class. used to delete assignment operator. */
class NonCopyable {
public:
    NonCopyable()                    = default;
    NonCopyable(const NonCopyable&)  = delete;
    NonCopyable(const NonCopyable&&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&&) = delete;
};
} // namespace MNN

#endif /* NonCopyable_hpp */
