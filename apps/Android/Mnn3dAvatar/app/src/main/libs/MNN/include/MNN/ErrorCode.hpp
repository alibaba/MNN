//
//  ErrorCode.hpp
//  MNN
//
//  Created by MNN on 2018/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_ErrorCode_h
#define MNN_ErrorCode_h

namespace MNN {
enum ErrorCode {
#ifdef NO_ERROR
#undef NO_ERROR
#endif // NO_ERROR
    NO_ERROR           = 0,
    OUT_OF_MEMORY      = 1,
    NOT_SUPPORT        = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION       = 4,
    INVALID_VALUE      = 5,

    // User error
    INPUT_DATA_ERROR = 10,
    CALL_BACK_STOP   = 11,

    // Op Resize Error
    TENSOR_NOT_SUPPORT = 20,
    TENSOR_NEED_DIVIDE = 21,

    // File error
    FILE_CREATE_FAILED = 30,
    FILE_REMOVE_FAILED = 31,
    FILE_OPEN_FAILED   = 32,
    FILE_CLOSE_FAILED  = 33,
    FILE_RESIZE_FAILED = 34,
    FILE_SEEK_FAILED   = 35,
    FILE_NOT_EXIST     = 36,
    FILE_UNMAP_FAILED  = 37
};
} // namespace MNN

#endif /* ErrorCode_h */
