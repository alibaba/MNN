//
//  VulkanFence.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanFence.hpp"
#if VK_FENCE_WAIT_FD_IF_SUPPORT
#include <errno.h>
#include <poll.h>
#include <string.h>
#include "MNNDefine.h"
#endif

namespace MNN {
VulkanFence::VulkanFence(const VulkanDevice& dev) : mDevice(dev) {
    CALL_VK(mDevice.createFence(mFence));
}
VulkanFence::~VulkanFence() {
    mDevice.destroyFence(mFence);
}

VkResult VulkanFence::rawWait() const {
    auto status = VK_TIMEOUT;
    do {
        status = mDevice.waitForFence(mFence, 5000000000);
    } while (status == VK_TIMEOUT);
    return status;
}

#if VK_FENCE_WAIT_FD_IF_SUPPORT
VkResult VulkanFence::pollWait(const int fd) const {
    struct pollfd fds;
    fds.events  = POLLIN | POLLERR;
    fds.fd      = fd;
    fds.revents = 0;
    int ret     = 0;
    int err     = 0;
    do {
        ret = poll(&fds, 1, -1);
        err = errno;
    } while ((ret < 0) && (EINTR == err));
    if (ret > 0) {
        if ((fds.revents & POLLIN) && (fds.fd == fd) && (!(fds.revents & POLLERR))) {
            return VK_SUCCESS;
        } else {
            MNN_ERROR("Fence Poll failed err=%d (%s)\n", err, strerror(err));
            return VK_ERROR_DEVICE_LOST;
        }
    } else {
        MNN_ERROR("Fence Poll failed err=%d (%s)\n", err, strerror(err));
        return VK_ERROR_DEVICE_LOST;
    }
    return VK_SUCCESS;
}

VkResult VulkanFence::fdWait() const {
    int fd       = 0;
    VkResult res = fenceFd(fd);
    if (VK_SUCCESS != res) {
        return rawWait();
    } else {
        // the special value -1 for fd is treated like a valid sync file descriptor referring to an object that has
        // already signaled. The import operation will succeed and the VkFence will have a temporarily imported payload
        // as if a valid file descriptor had been provided.
        if (-1 == fd) {
            return VK_SUCCESS;
        }
        return pollWait(fd);
    }
    return VK_SUCCESS;
}
#endif

VkResult VulkanFence::wait() const {
#if VK_FENCE_WAIT_FD_IF_SUPPORT
    if (supportFenceFd()) {
        return fdWait();
    }
    return rawWait();
#else
    return rawWait();
#endif
}
VkResult VulkanFence::reset() const {
    return mDevice.resetFence(mFence);
}
// if fenceFd is support, we can use epoll or select wait for fence complete
bool VulkanFence::supportFenceFd() const {
    return mDevice.supportFenceFd();
}

#ifdef VK_USE_PLATFORM_WIN32_KHR
VkResult VulkanFence::fenceFd(HANDLE& fd) const {
#else
VkResult VulkanFence::fenceFd(int& fd) const {
    return mDevice.fenceFd(mFence, fd);
#endif
}

} // namespace MNN
