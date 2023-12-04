//
////  commonKit.hpp
////
////  Created by MNN on 2023/11/03.
////  Copyright © 2018, Alibaba Group Holding Limited
////
//
#ifndef COMMONKit_HPP
#define COMMONKit_HPP

#include <iostream>
#include "MNN_compression.pb.h"
class CommonKit {
public:
    static bool FileIsExist(std::string path);
    static bool json2protobuf(const char* jsonFile, const char* protoFile=nullptr, MNN::Compression::Pipeline* pipeline=nullptr);
};
#endif
