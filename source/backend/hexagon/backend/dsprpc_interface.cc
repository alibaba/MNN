#include "dsprpc_interface.h"

#include <MNN/MNNDefine.h>
#include <dlfcn.h>

#include <mutex>

namespace {

std::mutex& dsprpcMutex() {
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<DspRpcInterface>& activeDspRpcInterface() {
    static std::shared_ptr<DspRpcInterface> interface;
    return interface;
}

std::shared_ptr<DspRpcInterface> acquireDspRpcInterface() {
    std::lock_guard<std::mutex> lock(dsprpcMutex());
    return activeDspRpcInterface();
}

}  // namespace

DspRpcInterface::DspRpcInterface() {
    mLib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
    if (mLib == nullptr) {
        MNN_ERROR("unable to load libcdsprpc.so");
        return;
    }
    mRpcmemInit = reinterpret_cast<decltype(mRpcmemInit)>(dlsym(mLib, "rpcmem_init"));
    mRpcmemDeinit = reinterpret_cast<decltype(mRpcmemDeinit)>(dlsym(mLib, "rpcmem_deinit"));
    mRpcmemAlloc = reinterpret_cast<decltype(mRpcmemAlloc)>(dlsym(mLib, "rpcmem_alloc"));
    mRpcmemFree = reinterpret_cast<decltype(mRpcmemFree)>(dlsym(mLib, "rpcmem_free"));
    mRpcmemToFd = reinterpret_cast<decltype(mRpcmemToFd)>(dlsym(mLib, "rpcmem_to_fd"));
    mRpcmemCacheFlush = reinterpret_cast<decltype(mRpcmemCacheFlush)>(dlsym(mLib, "rpcmem_cache_flush"));
    mRpcmemCacheInvalidate = reinterpret_cast<decltype(mRpcmemCacheInvalidate)>(dlsym(mLib, "rpcmem_cache_invalidate"));
    mFastrpcMmap = reinterpret_cast<decltype(mFastrpcMmap)>(dlsym(mLib, "fastrpc_mmap"));
    mFastrpcMunmap = reinterpret_cast<decltype(mFastrpcMunmap)>(dlsym(mLib, "fastrpc_munmap"));
}

DspRpcInterface::~DspRpcInterface() {
    if (mLib != nullptr) {
        MNN_PRINT("[MNN::Hexagon] dlclose libcdsprpc.so\n");
        dlclose(mLib);
        mLib = nullptr;
    }
}

bool DspRpcInterface::valid() const { return mLib != nullptr; }

void DspRpcInterface::rpcmemInit() const {
    if (mRpcmemInit != nullptr) {
        mRpcmemInit();
    }
}

void DspRpcInterface::rpcmemDeinit() const {
    if (mRpcmemDeinit != nullptr) {
        mRpcmemDeinit();
    }
}

void* DspRpcInterface::rpcmemAlloc(int heapId, uint32_t flags, int size) const {
    return mRpcmemAlloc != nullptr ? mRpcmemAlloc(heapId, flags, size) : nullptr;
}

void DspRpcInterface::rpcmemFree(void* p) const {
    if (mRpcmemFree != nullptr) {
        mRpcmemFree(p);
    }
}

int DspRpcInterface::rpcmemToFd(void* p) const { return mRpcmemToFd != nullptr ? mRpcmemToFd(p) : -1; }

int DspRpcInterface::rpcmemCacheFlush(void* po, int size) const {
    return mRpcmemCacheFlush != nullptr ? mRpcmemCacheFlush(po, size) : -1;
}

int DspRpcInterface::rpcmemCacheInvalidate(void* po, int size) const {
    return mRpcmemCacheInvalidate != nullptr ? mRpcmemCacheInvalidate(po, size) : -1;
}

int DspRpcInterface::fastrpcMmap(int domain, int fd, void* addr, int offset, size_t length,
                                 enum fastrpc_map_flags flags) const {
    return mFastrpcMmap != nullptr ? mFastrpcMmap(domain, fd, addr, offset, length, flags) : -1;
}

int DspRpcInterface::fastrpcMunmap(int domain, int fd, void* addr, size_t length) const {
    return mFastrpcMunmap != nullptr ? mFastrpcMunmap(domain, fd, addr, length) : -1;
}

void dsprpc_interface_set_active(const std::shared_ptr<DspRpcInterface>& interface) {
    std::lock_guard<std::mutex> lock(dsprpcMutex());
    activeDspRpcInterface() = interface;
}

void dsprpc_interface_clear_active(const DspRpcInterface* interface) {
    std::lock_guard<std::mutex> lock(dsprpcMutex());
    auto& active = activeDspRpcInterface();
    if (active.get() == interface) {
        active.reset();
    }
}

extern "C" {

void rpcmem_init(void) {
    auto interface = acquireDspRpcInterface();
    if (interface) {
        interface->rpcmemInit();
    }
}

void rpcmem_deinit(void) {
    auto interface = acquireDspRpcInterface();
    if (interface) {
        interface->rpcmemDeinit();
    }
}

void* rpcmem_alloc(int heap_id, uint32_t flags, int size) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->rpcmemAlloc(heap_id, flags, size) : nullptr;
}

void rpcmem_free(void* p) {
    auto interface = acquireDspRpcInterface();
    if (interface) {
        interface->rpcmemFree(p);
    }
}

int rpcmem_to_fd(void* p) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->rpcmemToFd(p) : -1;
}

int rpcmem_cache_flush(void* po, int size) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->rpcmemCacheFlush(po, size) : -1;
}

int rpcmem_cache_invalidate(void* po, int size) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->rpcmemCacheInvalidate(po, size) : -1;
}

int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->fastrpcMmap(domain, fd, addr, offset, length, flags) : -1;
}

int fastrpc_munmap(int domain, int fd, void* addr, size_t length) {
    auto interface = acquireDspRpcInterface();
    return interface ? interface->fastrpcMunmap(domain, fd, addr, length) : -1;
}

}  // extern "C"
