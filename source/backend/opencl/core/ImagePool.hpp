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
#include "core/NonCopyable.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
namespace MNN {
namespace OpenCL {

class ImagePool : public NonCopyable {
public:
    ImagePool(cl::Context& context) : mContext(context) {
    }

    cl::Image* alloc(int w, int h, cl_channel_type type, bool separate = false);
    void recycle(cl::Image* image, bool release = false);
    void clear();
    void releaseFreeList();

    struct Node {
        int w;
        int h;
        cl_channel_type type;
        std::shared_ptr<cl::Image> image;
    };

private:
    std::map<cl::Image*, std::shared_ptr<Node>> mAllImage;
    std::list<std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ImagePool_hpp */
