//
//  ImagePool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/ImagePool.hpp"
namespace MNN {
namespace OpenCL {
cl::Image* ImagePool::alloc(int w, int h, cl_channel_type type, bool separate) {
    if (!separate) {
        int minWaste  = 0;
        auto findIter = mFreeList.end();
        for (auto iterP = mFreeList.begin(); iterP != mFreeList.end(); iterP++) {
            auto& iter = *iterP;
            if (iter->w >= w && iter->h >= h && iter->type == type) {
                int waste = iter->w * iter->h - w * h;
                if (minWaste == 0 || waste < minWaste) {
                    findIter = iterP;
                    minWaste = waste;
                }
            }
        }
        if (findIter != mFreeList.end()) {
            auto image = (*findIter)->image.get();
            mFreeList.erase(findIter);
            return image;
        }
    }
    std::shared_ptr<Node> node(new Node);
    cl_int ret = CL_SUCCESS;
    node->w = w;
    node->h = h;
    node->type = type;
    node->image.reset(
        new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, type), w, h, 0, nullptr, &ret));
    if (nullptr == node->image.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Image %d x %d error, code:%d \n", w, h, ret);
        return nullptr;
    }
    mAllImage.insert(std::make_pair(node->image.get(), node));
    return node->image.get();
}

void ImagePool::recycle(cl::Image* image, bool release) {
    auto iter = mAllImage.find(image);
    if (iter == mAllImage.end()) {
        MNN_ERROR("recycle failed for not belong image\n");
        return;
    }
    if (release) {
        mAllImage.erase(iter);
        return;
    }
    mFreeList.push_back(iter->second);
}

void ImagePool::clear() {
    mFreeList.clear();
    mAllImage.clear();
}

void ImagePool::releaseFreeList() {
    for(auto mf : mFreeList){
        auto iter = mAllImage.find(mf->image.get());
        if (iter != mAllImage.end()) {
            mAllImage.erase(iter);
        }
    }
    mFreeList.clear();
}
} // namespace OpenCL
} // namespace MNN
