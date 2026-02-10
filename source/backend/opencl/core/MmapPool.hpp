//
//  MmapPool.hpp
//  MNN
//
//  Created by MNN on 2025/12/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MmapPool_hpp
#define MmapPool_hpp

#include <list>
#include <set>
#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include "core/BufferAllocator.hpp"
#include "core/MNNFileUtils.h"

namespace MNN {
namespace OpenCL {
struct OpenCLMmapBufferNode{
    OpenCLMmapBufferNode(){};
    std::string fileName;
    size_t offset;
    size_t size;
    std::shared_ptr<cl::Buffer> buffer;
};

struct OpenCLMmapImageNode {
    OpenCLMmapImageNode(){};
    std::string fileName;
    size_t offset;
    size_t size;
    int w;
    int h;
    cl_channel_type type;
    std::shared_ptr<cl::Image> image;
};

class OpenCLMmapAllocator {
private:
    std::map<std::string, std::tuple<file_t, size_t>> mCache;
    std::string mFileName;
    std::string mPrefix;
    std::string mPosfix;
    int mAllocTimes = 0;
    bool mRemove;
    bool mNewMmap = false;
    
public:
    OpenCLMmapAllocator(const char* dirName, const char* prefix, const char* posfix, bool autoRemove);
    ~ OpenCLMmapAllocator() {
        for (auto& iter : mCache) {
            MNNCloseFile(std::get<0>(iter.second));
            if (mRemove) {
                MNNRemoveFile(iter.first.c_str());
            }
        }
    }
    std::string onAlloc(size_t size);
    bool read(std::string fileName, size_t offset, size_t size, void* buffer);
    bool write(std::string fileName, size_t offset, size_t size, void* buffer);
    void sync();
};

class MmapPool : public NonCopyable {
public:
    MmapPool(std::shared_ptr<OpenCLMmapAllocator> origin, cl::Context& context, cl::CommandQueue& command, cl_mem_flags flags, int useCacheMmap) : mOrigin(origin), mContext(context), mCommand(command), mFlag(flags), mUseCachedMmap(useCacheMmap) {}
    
    std::shared_ptr<cl::Buffer> allocBuffer(size_t size, bool separate = false);
    std::shared_ptr<cl::Image> allocImage(size_t w, size_t h, cl_channel_type type, bool separate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void recycle(cl::Image* image, bool release = false);
    void clear();
    void releaseFreeList();
    void sync();
    size_t totalSize() { return mTotalSize; }

private:
    std::map<cl::Buffer*, std::shared_ptr<OpenCLMmapBufferNode>> mAllBuffer;
    std::multimap<size_t, std::shared_ptr<OpenCLMmapBufferNode>> mFreeBufferList;
    std::map<cl::Image*, std::shared_ptr<OpenCLMmapImageNode>> mAllImage;
    std::list<std::shared_ptr<OpenCLMmapImageNode>> mFreeImageList;
    std::map<std::string, size_t> mFileInfo;
    std::shared_ptr<OpenCLMmapAllocator> mOrigin;

    cl::Context& mContext;
    cl::CommandQueue& mCommand;
    cl_mem_flags mFlag;
    size_t mTotalSize = 0;
    int mUseCachedMmap;
    bool mHasSync = false;
    size_t mFileSize = 1024*1024*1024;
};


} // namespace OpenCL
} // namespace MNN

#endif /* MmapPool_hpp */
