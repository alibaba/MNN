//
//  OnnxUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ONNXUTILS_HPP
#define ONNXUTILS_HPP

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

bool onnx_read_proto_from_binary(const char* filepath, google::protobuf::Message* message);
bool onnx_write_proto_from_binary(const char* filepath, const google::protobuf::Message* message);

#endif // ONNXUTILS_HPP
