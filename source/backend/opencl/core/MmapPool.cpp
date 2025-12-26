//
//  MmapPool.cpp
//  MNN
//
//  Created by MNN on 2025/12/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/MmapPool.hpp"
namespace MNN {
namespace OpenCL {
// only support static memory
OpenCLMmapAllocator::OpenCLMmapAllocator(const char* dirName, const char* prefix, const char* posfix, bool autoRemove) {
    if (nullptr != dirName) {
        mFileName = dirName;
        if (!MNNCreateDir(dirName)) {
            MNN_ERROR("%s not exist\n", dirName);
        }
    }
    if (nullptr != prefix) {
        mPrefix = prefix;
    }
    if (nullptr != posfix) {
        mPosfix = posfix;
    }
    mRemove = autoRemove;
}

std::string OpenCLMmapAllocator::onAlloc(size_t size) {
    MNN_ASSERT(size > 0);
    MNN_ASSERT(!mSynced);
    std::string name = mPrefix + std::to_string(mAllocTimes) + "." + mPosfix;
    std::string fileName = MNNFilePathConcat(mFileName, name);
    file_t file;
    if (MNNFileExist(fileName.c_str())) {
        file = MNNOpenFile(fileName.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
    } else {
        file = MNNCreateFile(fileName.c_str());
        auto code = MNNSetFileSize(file, size);
        if (NO_ERROR != code) {
            MNN_ERROR("Set File size %lu error= %d\n", size, code);
        }
        mNewMmap = true;
    }
    mCache.insert(std::make_pair(fileName, std::make_tuple(file, size)));
    mAllocTimes++;
    return fileName;
}
bool OpenCLMmapAllocator::read(std::string fileName, size_t offset, size_t size, void* buffer){
    auto iter = mCache.find(fileName);
    if (iter == mCache.end()) {
        MNN_ASSERT(false);
        MNN_ERROR("Invalid mmap for OpenCLMmapAllocator\n");
        return false;
    }
    file_t file = std::get<0>(iter->second);
    auto ret = MNNSetFilePointer(file, offset);
    if (ret != NO_ERROR) {
        return false;
    }
    auto readSize = MNNReadFile(file, buffer, size);
    if (readSize != size) {
        return false;
    }
}
bool OpenCLMmapAllocator::write(std::string fileName, size_t offset, size_t size, void* buffer){
    auto iter = mCache.find(fileName);
    if (iter == mCache.end()) {
        MNN_ASSERT(false);
        MNN_ERROR("Invalid unMmap for OpenCLMmapAllocator\n");
        return false;
    }
    file_t file = std::get<0>(iter->second);
    auto ret = MNNSetFilePointer(file, offset);
    if (ret != NO_ERROR) {
        return false;
    }
    auto writeSize = MNNWriteFile(file, buffer, size);
    if (writeSize != size) {
        return false;
    }
}
void OpenCLMmapAllocator::sync() {
    if (!mRemove && mNewMmap) {
        std::string cacheName = mPrefix + "sync." + mPosfix;
        std::string fileName = MNNFilePathConcat(mFileName, cacheName);
        MNNCreateFile(fileName.c_str());
    }
}

std::shared_ptr<cl::Buffer> MmapPool::allocBuffer(size_t size, bool separate) {
    if (!separate) {
        auto iter = mFreeBufferList.lower_bound(size);
        if (iter != mFreeBufferList.end()) {
            auto buffer = iter->second->buffer;
            mFreeBufferList.erase(iter);
            return buffer;
        }
    }
    std::string fileName;
    for(auto iter : mFileInfo){
        if(mFileSize - iter.second >= size){
            fileName = iter.first;
        }
    }
    if(fileName.length() == 0){
        //need open new file
        fileName = mOrigin->onAlloc(mFileSize);
        mFileInfo.insert(std::make_pair(fileName, 0));
    }
    
    std::shared_ptr<OpenCLMmapBufferNode> node(new OpenCLMmapBufferNode);
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;
    node->fileName = fileName;
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
    node->offset = mFileInfo[fileName];
    mFileInfo[fileName] += size;
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
        return nullptr;
    }
    if(mUseCachedMmap > 1){
        auto CLptr = mCommand.enqueueMapBuffer(*node->buffer.get(), CL_TRUE, CL_MAP_WRITE, 0, node->size);
        if(CLptr == nullptr){
            MNN_ERROR("map buffer %d error\n", node->size);
            return nullptr;
        }
        mOrigin->read(fileName, node->offset, node->size, CLptr);
        mCommand.enqueueUnmapMemObject(*node->buffer.get(), CLptr);
    }
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));
    return node->buffer;
}

std::shared_ptr<cl::Image> MmapPool::allocImage(size_t w, size_t h, cl_channel_type type, bool separate) {
    if (!separate) {
        int minWaste  = 0;
        auto findIter = mFreeImageList.end();
        for (auto iterP = mFreeImageList.begin(); iterP != mFreeImageList.end(); iterP++) {
            auto& iter = *iterP;
            if (iter->w >= w && iter->h >= h && iter->type == type) {
                int waste = iter->w * iter->h - w * h;
                if (minWaste == 0 || waste < minWaste) {
                    findIter = iterP;
                    minWaste = waste;
                }
            }
        }
        if (findIter != mFreeImageList.end()) {
            auto image = (*findIter)->image;
            mFreeImageList.erase(findIter);
            return image;
        }
    }
    
    std::shared_ptr<OpenCLMmapImageNode> node(new OpenCLMmapImageNode);
    cl_int ret = CL_SUCCESS;
    size_t row_pitch, slice_pitch;
    node->w = w;
    node->h = h;
    node->type = type;
    node->image.reset(new cl::Image2D(mContext, mFlag, cl::ImageFormat(CL_RGBA, type), w, h, 0, nullptr, &ret));
    if (nullptr == node->image.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Image %d x %d error, code:%d \n", w, h, ret);
        return nullptr;
    }
    auto CLptr = mCommand.enqueueMapImage(*node->image.get(), CL_TRUE, CL_MAP_WRITE, {0, 0, 0}, {w, h, 1}, &row_pitch, &slice_pitch);
    if(CLptr == nullptr){
        MNN_ERROR("map Image %d x %d error\n", w, h);
        return nullptr;
    }
    size_t size = h * row_pitch;
    
    std::string fileName;
    for(auto iter : mFileInfo){
        if(mFileSize - iter.second >= size){
            fileName = iter.first;
        }
    }
    if(fileName.length() == 0){
        //need open new file
        fileName = mOrigin->onAlloc(mFileSize);
        mFileInfo.insert(std::make_pair(fileName, 0));
    }
    node->fileName = fileName;
    node->size = size;
    node->offset = mFileInfo[fileName];
    mFileInfo[fileName] += size;
    if(mUseCachedMmap > 1){
        mOrigin->read(fileName, node->offset, node->size, CLptr);
    }
    
    mCommand.enqueueUnmapMemObject(*node->image.get(), CLptr);
    mAllImage.insert(std::make_pair(node->image.get(), node));
    return node->image;
}

void MmapPool::recycle(cl::Buffer* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(iter);
        return;
    }
    mFreeBufferList.insert(std::make_pair(iter->second->size, iter->second));
}

void MmapPool::recycle(cl::Image* image, bool release) {
    auto iter = mAllImage.find(image);
    if (iter == mAllImage.end()) {
        MNN_ERROR("Error for recycle image\n");
        return;
    }
    if (release) {
        mAllImage.erase(iter);
        return;
    }
    mFreeImageList.push_back(iter->second);
}

void MmapPool::clear() {
    mFreeBufferList.clear();
    mFreeImageList.clear();
    mAllBuffer.clear();
    mAllImage.clear();
    mTotalSize = 0;
}

void MmapPool::releaseFreeList() {
    for(auto mf : mFreeBufferList){
        auto iter = mAllBuffer.find(mf.second->buffer.get());
        if (iter != mAllBuffer.end()) {
            mAllBuffer.erase(iter);
        }
    }
    mFreeBufferList.clear();
    
    for(auto mf : mFreeImageList){
        auto iter = mAllImage.find(mf->image.get());
        if (iter != mAllImage.end()) {
            mAllImage.erase(iter);
        }
    }
    mFreeImageList.clear();
}

void MmapPool::sync() {
    if(mHasSync){
        return;
    }
    if(mUseCachedMmap == 1){
        for(auto iter : mAllBuffer){
            auto node = iter.second;
            auto CLptr = mCommand.enqueueMapBuffer(*node->buffer.get(), CL_TRUE, CL_MAP_WRITE, 0, node->size);
            if(CLptr == nullptr){
                MNN_ERROR("map buffer %d error\n", node->size);
                continue;
            }
            mOrigin->write(node->fileName, node->offset, node->size, CLptr);
            mCommand.enqueueUnmapMemObject(*node->buffer.get(), CLptr);
        }
        for(auto iter : mAllImage){
            auto node = iter.second;
            size_t row_pitch, slice_pitch;
            size_t w = node->w;
            size_t h = node->h;
            auto CLptr = mCommand.enqueueMapImage(*node->image.get(), CL_TRUE, CL_MAP_WRITE, {0, 0, 0}, {w, h, 1}, &row_pitch, &slice_pitch);
            mOrigin->write(node->fileName, node->offset, node->size, CLptr);
            mCommand.enqueueUnmapMemObject(*node->image.get(), CLptr);
        }
    }
    mOrigin->sync();
    mHasSync = true;
};

} // namespace OpenCL
} // namespace MNN
