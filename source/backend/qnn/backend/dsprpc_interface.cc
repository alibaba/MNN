#include "dsprpc_interface.h"

#include <dlfcn.h>

#include <MNN/MNNDefine.h>

using rpcmem_init_t   = decltype(rpcmem_init);
using rpcmem_deinit_t = decltype(rpcmem_deinit);
using rpcmem_alloc_t  = decltype(rpcmem_alloc);
using rpcmem_free_t   = decltype(rpcmem_free);
using rpcmem_to_fd_t  = decltype(rpcmem_to_fd);

using fastrpc_mmap_t   = decltype(fastrpc_mmap);
using fastrpc_munmap_t = decltype(fastrpc_munmap);

namespace {

void * load_lib() {
    auto * lib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        MNN_ERROR("unable to load libcdsprpc.so");
    }
    return lib;
}

void * load_fn(const char * fn_name) {
    static void * lib = load_lib();
    return dlsym(lib, fn_name);
}

struct dsprpc_interface {
    rpcmem_init_t *   rpcmem_init_fn   = reinterpret_cast<rpcmem_init_t *>(load_fn("rpcmem_init"));
    rpcmem_deinit_t * rpcmem_deinit_fn = reinterpret_cast<rpcmem_deinit_t *>(load_fn("rpcmem_deinit"));
    rpcmem_alloc_t *  rpcmem_alloc_fn  = reinterpret_cast<rpcmem_alloc_t *>(load_fn("rpcmem_alloc"));
    rpcmem_free_t *   rpcmem_free_fn   = reinterpret_cast<rpcmem_free_t *>(load_fn("rpcmem_free"));
    rpcmem_to_fd_t *  rpcmem_to_fd_fn  = reinterpret_cast<rpcmem_to_fd_t *>(load_fn("rpcmem_to_fd"));

    fastrpc_mmap_t *   fastrpc_mmap_fn   = reinterpret_cast<fastrpc_mmap_t *>(load_fn("fastrpc_mmap"));
    fastrpc_munmap_t * fastrpc_munmap_fn = reinterpret_cast<fastrpc_munmap_t *>(load_fn("fastrpc_munmap"));

    static dsprpc_interface * instance() {
        static dsprpc_interface * _instance = new dsprpc_interface;
        return _instance;
    }
};

}  // namespace

extern "C" {

void rpcmem_init(void) {
    auto fn = dsprpc_interface::instance()->rpcmem_init_fn;
    if (fn) {
        fn();
    }
}

void rpcmem_deinit(void) {
    auto fn = dsprpc_interface::instance()->rpcmem_deinit_fn;
    if (fn) {
        fn();
    }
}

void * rpcmem_alloc(int heap_id, uint32_t flags, int size) {
    auto fn = dsprpc_interface::instance()->rpcmem_alloc_fn;
    return fn ? fn(heap_id, flags, size) : nullptr;
}

void rpcmem_free(void * p) {
    auto fn = dsprpc_interface::instance()->rpcmem_free_fn;
    if (fn) {
        fn(p);
    }
}

int rpcmem_to_fd(void * p) {
    auto fn = dsprpc_interface::instance()->rpcmem_to_fd_fn;
    return fn ? fn(p) : -1;
}

int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    auto fn = dsprpc_interface::instance()->fastrpc_mmap_fn;
    return fn ? fn(domain, fd, addr, offset, length, flags) : -1;
}

int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
    auto fn = dsprpc_interface::instance()->fastrpc_munmap_fn;
    return fn ? fn(domain, fd, addr, length) : -1;
}

}  // extern "C"
