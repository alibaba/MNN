#include "CPUResizeCache.hpp"
#include "../../core/TensorUtils.hpp"

namespace MNN {
Tensor* CPUResizeCache::findCacheTensor(const Tensor* src, MNN_DATA_FORMAT format) const {
    auto iter = mFormatCache.find(std::make_pair(src, format));
    if (iter == mFormatCache.end()) {
        return nullptr;
    }
    return iter->second.get();
}

void CPUResizeCache::pushCacheTensor(std::shared_ptr<Tensor> dst, const Tensor* src, MNN_DATA_FORMAT format) {
    mFormatCache.insert(std::make_pair(std::make_pair(src, format), dst));
}
void CPUResizeCache::reset() {
    mFormatCache.clear();
}
void CPUResizeCache::release() {
    for (auto iter : mFormatCache) {
        TensorUtils::getDescribeOrigin(iter.second.get())->mem = nullptr;
    }
}
};
