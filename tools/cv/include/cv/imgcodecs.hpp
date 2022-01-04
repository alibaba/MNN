//
//  imgcodecs.hpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IMGCODECS_HPP
#define IMGCODECS_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>

namespace MNN {
namespace CV {
using namespace Express;

enum ImreadModes {
    IMREAD_GRAYSCALE            = 0, // uint8_t gray
    IMREAD_COLOR                = 1, // uint8_t bgr
    IMREAD_ANYDEPTH             = 4, // float bgr
};

enum ImwriteFlags {
    IMWRITE_JPEG_QUALITY        = 1, // jpg, default is 95
};

MNN_PUBLIC bool haveImageReader(const std::string& filename);

MNN_PUBLIC bool haveImageWriter(const std::string& filename);

MNN_PUBLIC VARP imdecode(const std::vector<uint8_t>& buf, int flags);

MNN_PUBLIC std::pair<bool, std::vector<uint8_t>> imencode(std::string ext, VARP img,
                                const std::vector<int>& params = std::vector<int>());

MNN_PUBLIC VARP imread(const std::string& filename, int flags = IMREAD_COLOR);

MNN_PUBLIC bool imwrite(const std::string& filename, VARP img,
                        const std::vector<int>& params = std::vector<int>());

} // CV
} // MNN
#endif // IMGCODECS_HPP
