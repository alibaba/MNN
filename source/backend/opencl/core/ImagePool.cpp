//
//  ImagePool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ImagePool.hpp"
namespace MNN {
namespace OpenCL {
cl::Image* ImagePool::alloc(int w, int h, bool seperate) {
    if (!seperate) {
        int minWaste  = 0;
        auto findIter = mFreeList.end();
        for (auto iterP = mFreeList.begin(); iterP != mFreeList.end(); iterP++) {
            auto& iter = *iterP;
            if (iter->w > w && iter->h > h) {
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
    node->w = w;
    node->h = h;
    node->image.reset(
        new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, mType), w, h, 0, nullptr, nullptr));
    if (nullptr == node->image) {
        MNN_ERROR("All Image %d x %d error \n", w, h);
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

} // namespace OpenCL
} // namespace MNN
