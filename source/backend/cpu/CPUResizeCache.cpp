#include "CPUResizeCache.hpp"
#include "../../core/TensorUtils.hpp"

namespace MNN {
std::shared_ptr<Tensor> CPUResizeCache::findCacheTensor(const Tensor* src, MNN_DATA_FORMAT format) const {
    auto iter = mFormatCache.find(std::make_pair(src, format));
    if (iter == mFormatCache.end()) {
        return nullptr;
    }
    return iter->second;
}

void CPUResizeCache::pushCacheTensor(std::shared_ptr<Tensor> dst, const Tensor* src, MNN_DATA_FORMAT format) {
    MNN_ASSERT(mFormatCache.find(std::make_pair(src, format)) == mFormatCache.end());
    mFormatCache.insert(std::make_pair(std::make_pair(src, format), dst));
}
void CPUResizeCache::reset() {
    mFormatCache.clear();
}
void CPUResizeCache::release() {
    // An entry must not be reused by a later resize: the hit is consumed without
    // re-acquiring the buffer, so once the arena is released the cached tensor may
    // end up bound to memory that belongs to somebody else.
    for (auto iter : mFormatCache) {
        TensorUtils::getDescribeOrigin(iter.second.get())->mem = nullptr;
    }
    mFormatCache.clear();
}
}; // namespace MNN
