//
//  draw.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#include "cv/types.hpp"
#include "cv/imgcodecs.hpp"
#include "cv/imgproc/color.hpp"
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STB_IMAGE_STATIC
#define STB_IMAGE_WRITE_STATIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace MNN {
namespace CV {

// helper functions
static VARP buildImgVARP(uint8_t* img, int height, int width, int channel, int flags) {
    auto rgb = _Const(img, {height, width, channel}, NHWC, halide_type_of<uint8_t>());
    free(img);
    VARP res;
    switch (flags) {
        case IMREAD_COLOR:
            res = cvtColor(rgb, COLOR_RGB2BGR);
            break;
        case IMREAD_GRAYSCALE:
            res = cvtColor(rgb, COLOR_RGB2GRAY);
            break;
        case IMREAD_ANYDEPTH:
            res = _Cast<float>(rgb);
            break;
        default:
            MNN_ERROR("Don't support imread flags!");
            return rgb;
    }
    return res;
}

static void writeFunc(void *context, void *data, int size) {
    std::vector<uint8_t>* ctx = (std::vector<uint8_t>*)context;
    ctx->insert(ctx->end(), (uint8_t*)data, (uint8_t*)data + size);
}
static std::string getExt(std::string name) {
    auto ext = name.substr(name.rfind('.') + 1, -1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}
// helper functions

bool haveImageReader(const std::string& filename) {
    int width, height, channel;
    return stbi_info(filename.c_str(), &width, &height, &channel);
}

bool haveImageWriter(const std::string& filename) {
    static const std::set<std::string> supportImages {
        "jpg", "jpeg", "png", "bmp", /* "gif", "psd", "pic", "pnm", "hdr", "tga" */
    };
    return supportImages.find(getExt(filename)) != supportImages.end();
}

VARP imdecode(const std::vector<uint8_t>& buf, int flags) {
    int width, height, channel;
    auto img = stbi_load_from_memory(buf.data(), buf.size(), &width, &height, &channel, 3);
    if (nullptr == img) {
        MNN_ERROR("Can't decode\n");
        return nullptr;
    }
    return buildImgVARP(img, height, width, 3, flags);
}

std::pair<bool, std::vector<uint8_t>> imencode(std::string ext, VARP img, const std::vector<int>& params) {
    VARP rgb = cvtColor(img, COLOR_BGR2RGB);
    int height, width, channel;
    getVARPSize(rgb, &height, &width, &channel);
    ext = getExt(ext);
    bool res = false;
    std::vector<uint8_t> buf;
    if (ext == "jpg" || "jpeg") {
        int quality = 95;
        for (size_t i = 0; i < params.size(); i += 2) {
            if (params[i] == IMWRITE_JPEG_QUALITY) {
                quality = params[i + 1];
                break;
            }
        }
        res = stbi_write_jpg_to_func(writeFunc, (void*)&buf, width, height, channel, rgb->readMap<uint8_t>(), quality);
    }
    if (ext == "png") {
        res = stbi_write_png_to_func(writeFunc, (void*)&buf, width, height, channel, rgb->readMap<uint8_t>(), 0);
    }
    if (ext == "bmp") {
        res = stbi_write_bmp_to_func(writeFunc, (void*)&buf, width, height, channel, rgb->readMap<uint8_t>());
    }
    return { res, buf };
}

VARP imread(const std::string& filename, int flags) {
    int width, height, channel;
    auto img = stbi_load(filename.c_str(), &width, &height, &channel, 3);
    if (nullptr == img) {
        MNN_ERROR("Can't open %s\n", filename.c_str());
        return nullptr;
    }
    return buildImgVARP(img, height, width, 3, flags);
}

bool imwrite(const std::string& filename, VARP img, const std::vector<int>& params) {
    if (img->getInfo()->type != halide_type_of<uint8_t>()) {
        img = _Cast<uint8_t>(img);
    }
    int height, width, channel;
    getVARPSize(img, &height, &width, &channel);
    if (channel == 3) {
        img = cvtColor(img, COLOR_BGR2RGB);
    } else {
        MNN_ERROR("MNN cv imwrite just support RGB/BGR format.");
    }
    auto ext = getExt(filename);
    if (ext == "jpg" || ext == "jpeg") {
        int quality = 95;
        for (size_t i = 0; i < params.size(); i += 2) {
            if (params[i] == IMWRITE_JPEG_QUALITY) {
                quality = params[i + 1];
                break;
            }
        }
        return stbi_write_jpg(filename.c_str(), width, height, channel, img->readMap<uint8_t>(), quality);
    }
    if (ext == "png") {
        return stbi_write_png(filename.c_str(), width, height, channel, img->readMap<uint8_t>(), 0);
    }
    if (ext == "bmp") {
        return stbi_write_bmp(filename.c_str(), width, height, channel, img->readMap<uint8_t>());
    }
    return false;
}

} // CV
} // MNN
