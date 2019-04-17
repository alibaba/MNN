//
//  CaffeUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CAFFEUTILS_HPP
#define CAFFEUTILS_HPP

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

bool read_proto_from_text(const char* filepath, google::protobuf::Message* message);
bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message);
#endif // CAFFEUTILS_HPP
