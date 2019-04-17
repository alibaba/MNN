//
//  ImagePool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImagePool_hpp
#define ImagePool_hpp

#include <list>
#include <map>
#include "NonCopyable.hpp"
#include "core/runtime/OpenCLWrapper.hpp"
namespace MNN {
namespace OpenCL {

class ImagePool : public NonCopyable {
public:
    ImagePool(cl::Context& context, cl_channel_type type) : mContext(context), mType(type) {
    }

    cl::Image* alloc(int w, int h, bool seperate = false);
    void recycle(cl::Image* image, bool release = false);
    void clear();

    struct Node {
        int w;
        int h;
        std::shared_ptr<cl::Image> image;
    };

private:
    std::map<cl::Image*, std::shared_ptr<Node>> mAllImage;
    std::list<std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_channel_type mType;
};
} // namespace OpenCL
} // namespace MNN
#endif /* ImagePool_hpp */
