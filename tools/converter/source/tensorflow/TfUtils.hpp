//
//  TfUtils.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TFUTILS_HPP
#define TFUTILS_HPP

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <functional>
#include "TmpGraph.hpp"
#include "graph.pb.h"

// import tensorflow GraphDef from file
bool tf_read_proto_from_binary(const char* filepath, google::protobuf::Message* message);

// get node's attribute according to the key
bool find_attr_value(const tensorflow::NodeDef* node, const char* key, tensorflow::AttrValue& value);

// Convert weight format from [KH,KW,CI,CO] to [CO,CI,KH,KW]
bool convertDataFormat(const float* src, float* dst, int planeNumber, int CI, int CO);


#endif // TFUTILS_HPP
