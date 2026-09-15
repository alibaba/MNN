//
//  CoreMLRaster.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLRaster.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "core/OpCommonUtils.hpp"

namespace MNN {

CoreMLRaster::CoreMLRaster(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

bool CoreMLRaster::buildReshape(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    mCoreMLBackend->setLayerName(layer, "Reshape");
    layer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESHAPE_STATIC;
    layer->reshapestatic = mCoreMLBackend->create<CoreML__Specification__ReshapeStaticLayerParams>();
    core_ml__specification__reshape_static_layer_params__init(layer->reshapestatic);
    auto outputShape = output->shape();
    // [coreml-rec-fix] Canonicalize to rank-4 [N,C,H,W]. CoreML
    // NeuralNetwork layers broadcast per-channel params on axis C of a
    // rank-3/4 feature; a rank-3 MNN tensor like [1,160,40] (N=1) used to
    // be emitted verbatim and the following scale/conv read C=1, then
    // broadcast a [160]-gamma into a 160x-too-large tensor (seen as
    // (160,160,40) shape explosions in PP-OCR rec's SE block). Appending
    // trailing 1s (and treating shape[0]==1 as N) keeps memory order and
    // restores C semantics.
    if (outputShape.size() < 4) {
        std::vector<int> canon;
        if (outputShape.size() >= 2 && outputShape[0] == 1) {
            // Leading singleton: it is N; next dim is C.
            canon.push_back(1);                       // N
            canon.push_back(outputShape[1]);          // C
            for (size_t k = 2; k < outputShape.size(); k++) canon.push_back(outputShape[k]);
        } else {
            // Empirical (PP-OCR rec, M4/macOS 26): treat dim0 as C with a
            // virtual N=1 — [48,960,1] becomes [1,48,960,1]. This is the
            // only form that produced correct predictions end-to-end
            // (checksum 39.996 vs CPU 40.000 at B=1). Keeping dim0 as N
            // or plain-appending both broke every batch size.
            canon.push_back(1);                       // N
            canon.push_back(outputShape.size() >= 1 ? outputShape[0] : 1); // C
            for (size_t k = 1; k < outputShape.size(); k++) canon.push_back(outputShape[k]);
        }
        while (canon.size() < 4) canon.push_back(1);
        outputShape = canon;
    }
    layer->reshapestatic->n_targetshape = outputShape.size();
    layer->reshapestatic->targetshape = mCoreMLBackend->create<int64_t>(layer->reshapestatic->n_targetshape);
    for (int i = 0; i < outputShape.size(); i++) {
        layer->reshapestatic->targetshape[i] = outputShape[i];
    }
     if (outputShape.size() == 0) {
         layer->reshapestatic->n_targetshape = 1;
         layer->reshapestatic->targetshape = mCoreMLBackend->create<int64_t>(layer->reshapestatic->n_targetshape);
         layer->reshapestatic->targetshape[0] = 1;
     }
    mCoreMLBackend->setLayerInputs(layer, {mCoreMLBackend->getTensorName(input)});
    return true;
}
bool CoreMLRaster::buildPermute(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    bool needReshape = input->dimensions() != output->dimensions();
    CoreML__Specification__NeuralNetworkLayer *permuteLayer = layer, *reshapeLayer = nullptr;
    if (needReshape) {
        permuteLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(permuteLayer);
        reshapeLayer = layer;
    }
    mCoreMLBackend->setLayerName(permuteLayer, "Transpose");
    // [coreml-rec-fix diagnostic]
    if (getenv("MNN_COREML_DEBUG_PERMUTE")) {
        auto si = input->shape(); auto so = output->shape();
        const auto& rg = TensorUtils::getDescribe(input)->regions;
        fprintf(stderr, "[permute] in=[%s] out=[%s] nreg=%zu", [&]{std::string s; for(auto d:si) s+=std::to_string(d)+","; return s;}().c_str(), [&]{std::string s; for(auto d:so) s+=std::to_string(d)+","; return s;}().c_str(), rg.size());
        if (rg.size() == 1) {
            fprintf(stderr, " size=[%d,%d,%d] src=[%d,%d,%d] dst=[%d,%d,%d]",
                    rg[0].size[0], rg[0].size[1], rg[0].size[2],
                    (int)rg[0].src.stride[0], (int)rg[0].src.stride[1], (int)rg[0].src.stride[2],
                    (int)rg[0].dst.stride[0], (int)rg[0].dst.stride[1], (int)rg[0].dst.stride[2]);
        }
        fprintf(stderr, " needReshape=%d\n", (int)needReshape);
    }
    permuteLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TRANSPOSE;
    permuteLayer->transpose = mCoreMLBackend->create<CoreML__Specification__TransposeLayerParams>();
    core_ml__specification__transpose_layer_params__init(permuteLayer->transpose);
    permuteLayer->transpose->n_axes = 4;
    permuteLayer->transpose->axes = mCoreMLBackend->create<uint64_t>(permuteLayer->transpose->n_axes);
    auto srcFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dstFormat = TensorUtils::getDescribe(output)->dimensionFormat;
    // [coreml-rec-fix] Stride-derived axes first, for every format pair:
    // the raster Region is the ground truth of the copy, the format
    // heuristics below only apply when no usable region exists. This
    // matters because rank-3 tensors carrying a NC4HW4/NHWC format tag
    // (SE blocks, squeeze/unsqueeze chains in recognition models) used
    // to hit the hardcoded rank-4 transposes and produced garbage axes.
    bool axesFromRegion = false;
    {
        const auto& outRegions = TensorUtils::getDescribe(output)->regions;
        const auto& regions = outRegions;
        if (regions.size() == 1 && regions[0].origin == input) {
            const auto& region = regions[0];
            int inDim = region.origin->dimensions();
            int outDim = output->dimensions();
            int effIn = inDim < 3 ? 3 : inDim;
            int effOut = outDim < 3 ? 3 : outDim;
            int rank = effIn > effOut ? effIn : effOut;
            if (rank <= 4) {
                const int kNoStride = -1;
                int srcStride[4], dstStride[4];
                for (int i = 0; i < 4; i++) {
                    srcStride[i] = (i < effIn - 3) ? kNoStride : -2;
                    dstStride[i] = (i < effOut - 3) ? kNoStride : -2;
                }
                for (int i = 0; i < 3; i++) {
                    int si = effIn - 3 + i;
                    int di = effOut - 3 + i;
                    int64_t ss = region.src.stride[i];
                    int64_t ds = region.dst.stride[i];
                    bool sSkip = (region.size[i] <= 1) || (ss == 0);
                    bool dSkip = (region.size[i] <= 1) || (ds == 0);
                    if (sSkip) ss = -3;
                    if (dSkip) ds = -3;
                    if (si >= 0) srcStride[si] = (int)ss;
                    if (di >= 0) dstStride[di] = (int)ds;
                }
                int inShapeEff[4] = {1, 1, 1, 1};
                int outShapeEff[4] = {1, 1, 1, 1};
                {
                    const auto ish = region.origin->shape();
                    for (int i = 0; i < inDim && i < 4; i++) inShapeEff[i] = ish[i];
                    const auto osh = output->shape();
                    for (int i = 0; i < outDim && i < 4; i++) outShapeEff[i] = osh[i];
                }
                // [coreml-rec-fix] The Transpose layer operates at the
                // INPUT's rank; any extra output dims are materialized by
                // the trailing reshapeStatic. So axes has effIn entries,
                // and matching walks INPUT axes j (outer->inner), finding
                // for each the destination slot by stride equality in the
                // region (dst enumerates the permuted order directly).
                int rankIn = effIn;
                int axes[4];
                bool ok = true;
                bool used[4] = {false, false, false, false};
                // Destination slot strides in OUTPUT enumeration order
                // (outer->inner of the region's dst loops):
                int dstSlot[3] = { (int)region.dst.stride[0],
                                  (int)region.dst.stride[1],
                                  (int)region.dst.stride[2] };
                int srcSlot[3] = { (int)region.src.stride[0],
                                  (int)region.src.stride[1],
                                  (int)region.src.stride[2] };
                int szSlot[3] = { region.size[0], region.size[1], region.size[2] };
                // Build the permutation as: for each input axis j (in the
                // input's own C-order position), which region slot walks
                // it contiguously? An input axis j is "slot k" iff the
                // slot covers exactly that axis (slot stride == C-order
                // stride of axis j in the origin, slot size == dim).
                int64_t inStrides[4] = {0,0,0,0};
                {
                    int64_t acc = 1;
                    for (int j = rankIn - 1; j >= 0; j--) {
                        inStrides[j] = acc;
                        acc *= inShapeEff[j];
                    }
                }
                for (int j = 0; j < rankIn && ok; j++) {
                    int slot = -1;
                    for (int k = 0; k < 3; k++) {
                        if (szSlot[k] == inShapeEff[j] && srcSlot[k] == (int)inStrides[j]) {
                            slot = k;
                            break;
                        }
                    }
                    if (slot < 0) {
                        // Singleton input axis: unconstrained; take the
                        // first unused slot with size 1, else slot 0.
                        for (int k = 0; k < 3; k++) {
                            if (!used[k] && szSlot[k] <= 1) { slot = k; break; }
                        }
                    }
                    if (slot < 0) slot = 0;
                    if (used[slot]) { ok = false; break; }
                    used[slot] = true;
                    // axes[slot] = j: output slot `slot` (in dst loop
                    // order, which IS the output C-order for the
                    // transposed rank-in tensor) draws from input axis j.
                    axes[slot] = j;
                }
                if (ok) {
                    // Sanity: must be a permutation of 0..rankIn-1.
                    bool seen[4] = {false,false,false,false};
                    for (int k = 0; k < rankIn; k++) {
                        if (axes[k] < 0 || axes[k] >= rankIn || seen[axes[k]]) { ok = false; break; }
                        seen[axes[k]] = true;
                    }
                }
                if (ok) {
                    permuteLayer->transpose->n_axes = rankIn;
                    for (int i = 0; i < rankIn; i++) {
                        permuteLayer->transpose->axes[i] = axes[i];
                    }
                    axesFromRegion = true;
                    if (getenv("MNN_COREML_DEBUG_PERMUTE")) {
                        fprintf(stderr, "[permute-axes] rankIn=%d axes=[%d,%d,%d,%d] sz=[%d,%d,%d] src=[%d,%d,%d] dst=[%d,%d,%d] in=[%d,%d,%d,%d] inStr=[%d,%d,%d,%d]\n",
                                rankIn, axes[0], axes[1], axes[2], axes[3],
                                szSlot[0], szSlot[1], szSlot[2],
                                srcSlot[0], srcSlot[1], srcSlot[2],
                                dstSlot[0], dstSlot[1], dstSlot[2],
                                inShapeEff[0], inShapeEff[1], inShapeEff[2], inShapeEff[3],
                                (int)inStrides[0], (int)inStrides[1], (int)inStrides[2], (int)inStrides[3]);
                    }
                }
            }
        }
    }
    // NCHW -> NHWC (fallback, rank-4 only)
    if (!axesFromRegion && (srcFormat == MNN_DATA_FORMAT_NC4HW4 || srcFormat == MNN_DATA_FORMAT_NCHW)
        && dstFormat == MNN_DATA_FORMAT_NHWC && input->dimensions() == 4) {
        permuteLayer->transpose->n_axes = 4;
        permuteLayer->transpose->axes[0] = 0;
        permuteLayer->transpose->axes[1] = 2;
        permuteLayer->transpose->axes[2] = 3;
        permuteLayer->transpose->axes[3] = 1;
    }
    // NHWC -> NCHW (fallback, rank-4 only)
    if (!axesFromRegion && (dstFormat == MNN_DATA_FORMAT_NC4HW4 || srcFormat == MNN_DATA_FORMAT_NCHW)
        && srcFormat == MNN_DATA_FORMAT_NHWC && input->dimensions() == 4) {
        permuteLayer->transpose->n_axes = 4;
        permuteLayer->transpose->axes[0] = 0;
        permuteLayer->transpose->axes[1] = 3;
        permuteLayer->transpose->axes[2] = 1;
        permuteLayer->transpose->axes[3] = 2;
    }
    if (!axesFromRegion) {
        // Last-resort fallback: legacy shape-value matching. Only reached
        // when no usable raster region exists. Clamped to the output rank
        // so no stale slots survive (axis 0 twice = invalid permutation).
        auto inputShape = input->shape();
        auto outputShape = output->shape();
        permuteLayer->transpose->n_axes = outputShape.size();
        for (int i = 0; i < outputShape.size(); i++) {
            auto dimVal = outputShape[i];
            auto axis = -1;
            for (int j = 0; j < inputShape.size(); j++) {
                if (inputShape[j] == dimVal) {
                    axis = j;
                    break;
                }
            }
            permuteLayer->transpose->axes[i] = axis;
        }
    }
    mCoreMLBackend->setLayerInputs(permuteLayer, {mCoreMLBackend->getTensorName(input)});
    if (reshapeLayer) {
        std::string middleName = mCoreMLBackend->getTensorName(input) + "_permute_" + mCoreMLBackend->getTensorName(output);
        mCoreMLBackend->setLayerOutputs(permuteLayer, {middleName});
        mCoreMLBackend->addLayer(permuteLayer);
        mCoreMLBackend->setLayerName(reshapeLayer, "Permute_Reshape");
        reshapeLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESHAPE_STATIC;
        reshapeLayer->reshapestatic = mCoreMLBackend->create<CoreML__Specification__ReshapeStaticLayerParams>();
        core_ml__specification__reshape_static_layer_params__init(reshapeLayer->reshapestatic);
        auto outputShape = output->shape();
        reshapeLayer->reshapestatic->n_targetshape = outputShape.size();
        reshapeLayer->reshapestatic->targetshape = mCoreMLBackend->create<int64_t>(reshapeLayer->reshapestatic->n_targetshape);
        for (int i = 0; i < outputShape.size(); i++) {
            reshapeLayer->reshapestatic->targetshape[i] = outputShape[i];
        }
        mCoreMLBackend->setLayerInputs(reshapeLayer, {middleName});
    }
    return true;
}

bool CoreMLRaster::buildPad(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    bool needPermute = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC;
    CoreML__Specification__NeuralNetworkLayer *padLayer = layer, *postPermute = nullptr;
    std::string inputName = mCoreMLBackend->getTensorName(input);
    if (needPermute) {
        padLayer = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(padLayer);
        postPermute = layer;
        // NHWC -> NCHW
        auto prePermute = mCoreMLBackend->create<CoreML__Specification__NeuralNetworkLayer>();
        core_ml__specification__neural_network_layer__init(prePermute);
        mCoreMLBackend->setLayerName(prePermute, "prePermute");
        prePermute->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PERMUTE;
        prePermute->permute = mCoreMLBackend->create<CoreML__Specification__PermuteLayerParams>();
        core_ml__specification__permute_layer_params__init(prePermute->permute);
        prePermute->permute->n_axis = 4;
        prePermute->permute->axis = mCoreMLBackend->create<uint64_t>(prePermute->permute->n_axis);
        prePermute->permute->axis[0] = 0;
        prePermute->permute->axis[1] = 3;
        prePermute->permute->axis[2] = 1;
        prePermute->permute->axis[3] = 2;
        setLayerInputsAndOutputs(prePermute, {inputName}, {inputName + "-permute"});
        inputName = inputName + "-permute";
        mCoreMLBackend->addLayer(prePermute);
    }
    int padh = output->height() - input->height(), padw = output->width() - input->width();
    int top = padh / 2, bottom = std::ceil(padh / 2.0), left = padw / 2, right = std::ceil(padw / 2.0);
    mCoreMLBackend->setLayerName(padLayer, "Pad");
    padLayer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PADDING;
    padLayer->padding = mCoreMLBackend->create<CoreML__Specification__PaddingLayerParams>();
    core_ml__specification__padding_layer_params__init(padLayer->padding);
    padLayer->padding->padding_type_case = CORE_ML__SPECIFICATION__PADDING_LAYER_PARAMS__PADDING_TYPE_CONSTANT;
    padLayer->padding->constant = mCoreMLBackend->create<CoreML__Specification__PaddingLayerParams__PaddingConstant>();
    core_ml__specification__padding_layer_params__padding_constant__init(padLayer->padding->constant);
    padLayer->padding->constant->value = 0;
    padLayer->padding->paddingamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts>();
    core_ml__specification__border_amounts__init(padLayer->padding->paddingamounts);
    padLayer->padding->paddingamounts->n_borderamounts = 2;
    padLayer->padding->paddingamounts->borderamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes*>(2);
    padLayer->padding->paddingamounts->borderamounts[0] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(padLayer->padding->paddingamounts->borderamounts[0]);
    padLayer->padding->paddingamounts->borderamounts[0]->startedgesize = top;
    padLayer->padding->paddingamounts->borderamounts[0]->endedgesize = bottom;
    padLayer->padding->paddingamounts->borderamounts[1] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(padLayer->padding->paddingamounts->borderamounts[1]);
    padLayer->padding->paddingamounts->borderamounts[1]->startedgesize = left;
    padLayer->padding->paddingamounts->borderamounts[1]->endedgesize = right;
    mCoreMLBackend->setLayerInputs(padLayer, {inputName});
    if (needPermute) {
        inputName = inputName + "-pad";
        mCoreMLBackend->setLayerOutputs(padLayer, {inputName});
        mCoreMLBackend->addLayer(padLayer);
        // NHWC -> NCHW
        mCoreMLBackend->setLayerName(postPermute, "postPermute");
        postPermute->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PERMUTE;
        postPermute->permute = mCoreMLBackend->create<CoreML__Specification__PermuteLayerParams>();
        core_ml__specification__permute_layer_params__init(postPermute->permute);
        postPermute->permute->n_axis = 4;
        postPermute->permute->axis = mCoreMLBackend->create<uint64_t>(postPermute->permute->n_axis);
        postPermute->permute->axis[0] = 0;
        postPermute->permute->axis[1] = 2;
        postPermute->permute->axis[2] = 3;
        postPermute->permute->axis[3] = 1;
        mCoreMLBackend->setLayerInputs(postPermute, {inputName});
    }
    return true;
}

bool CoreMLRaster::buildCrop(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        return false;
    }
    int croph = input->height() - output->height(), cropw = input->width() - output->width();
    int top = croph / 2, bottom = std::ceil(croph / 2.0), left = cropw / 2, right = std::ceil(cropw / 2.0);
    mCoreMLBackend->setLayerName(layer, "Crop");
    layer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CROP;
    layer->crop = mCoreMLBackend->create<CoreML__Specification__CropLayerParams>();
    core_ml__specification__crop_layer_params__init(layer->crop);
    layer->crop->cropamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts>();
    core_ml__specification__border_amounts__init(layer->padding->paddingamounts);
    layer->crop->cropamounts->n_borderamounts = 2;
    layer->crop->cropamounts->borderamounts = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes*>(2);
    layer->crop->cropamounts->borderamounts[0] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(layer->crop->cropamounts->borderamounts[0]);
    layer->crop->cropamounts->borderamounts[0]->startedgesize = top;
    layer->crop->cropamounts->borderamounts[0]->endedgesize = bottom;
    layer->crop->cropamounts->borderamounts[1] = mCoreMLBackend->create<CoreML__Specification__BorderAmounts__EdgeSizes>();
    core_ml__specification__border_amounts__edge_sizes__init(layer->crop->cropamounts->borderamounts[1]);
    layer->crop->cropamounts->borderamounts[1]->startedgesize = left;
    layer->crop->cropamounts->borderamounts[1]->endedgesize = right;
    mCoreMLBackend->setLayerInputs(layer, {mCoreMLBackend->getTensorName(input)});
    return true;
}

bool CoreMLRaster::buildSlice(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    int endc = output->channel(), endh = output->height(), endw = output->width();
    bool maskc = endc == input->channel(), maskh = endh == input->height(), maskw = endw == input->width();
    mCoreMLBackend->setLayerName(layer, "Slice");
    layer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SLICE_STATIC;
    layer->slicestatic = mCoreMLBackend->create<CoreML__Specification__SliceStaticLayerParams>();
    core_ml__specification__slice_static_layer_params__init(layer->slicestatic);
    // [Seq, N, C, H, W] : [0:1:-1, 0:1:-1, 0:1:endc, 0:1:endh, 0:1:endw]
    int dims = 5;
    layer->slicestatic->n_beginids = dims;
    layer->slicestatic->beginids = mCoreMLBackend->create<int64_t>(dims);
    layer->slicestatic->n_beginmasks = dims;
    layer->slicestatic->beginmasks = mCoreMLBackend->create<int>(dims);
    layer->slicestatic->n_strides = dims;
    layer->slicestatic->strides = mCoreMLBackend->create<int64_t>(dims);
    for (int i = 0; i < dims; i++) {
        layer->slicestatic->beginids[i] = 0;
        layer->slicestatic->beginmasks[i] = true;
        layer->slicestatic->strides[i] = 1;
    }
    layer->slicestatic->n_endids = dims;
    layer->slicestatic->endids = mCoreMLBackend->create<int64_t>(dims);
    layer->slicestatic->n_endmasks = dims;
    layer->slicestatic->endmasks = mCoreMLBackend->create<int>(dims);
    layer->slicestatic->endids[0] = -1;
    layer->slicestatic->endids[1] = -1;
    layer->slicestatic->endids[2] = endc;
    layer->slicestatic->endids[3] = endh;
    layer->slicestatic->endids[4] = endw;
    layer->slicestatic->endmasks[0] = true;
    layer->slicestatic->endmasks[1] = true;
    layer->slicestatic->endmasks[2] = maskc;
    layer->slicestatic->endmasks[3] = maskh;
    layer->slicestatic->endmasks[4] = maskw;
    mCoreMLBackend->setLayerInputs(layer, {mCoreMLBackend->getTensorName(input)});
    return true;
}

bool CoreMLRaster::buildDepthToSpace(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output) {
    int blockSize = output->height() / input->height();
    mCoreMLBackend->setLayerName(layer, "DepthToSpace");
    layer->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_REORGANIZE_DATA;
    layer->reorganizedata = mCoreMLBackend->create<CoreML__Specification__ReorganizeDataLayerParams>();
    core_ml__specification__reorganize_data_layer_params__init(layer->reorganizedata);
    layer->reorganizedata->blocksize = blockSize;
    // layer->reorganizedata->mode = CORE_ML__SPECIFICATION__REORGANIZE_DATA_LAYER_PARAMS__REORGANIZATION_TYPE__DEPTH_TO_SPACE;
    layer->reorganizedata->mode = CORE_ML__SPECIFICATION__REORGANIZE_DATA_LAYER_PARAMS__REORGANIZATION_TYPE__PIXEL_SHUFFLE;
    mCoreMLBackend->setLayerInputs(layer, {mCoreMLBackend->getTensorName(input)});
    return true;
}

bool CoreMLRaster::rasterOptimization(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto& regions = TensorUtils::getDescribe(inputs[0])->regions;
    const auto region = regions[0];
    // region_size = 1: reshape, transpose
    if (regions.size() == 1) {
        int inputSize = 1, outputSize = 1;
        for (int i = 0; i < region.origin->dimensions(); i++) {
            inputSize *= region.origin->length(i);
        }
        for (int i = 0; i < outputs[0]->dimensions(); i++) {
            outputSize *= outputs[0]->length(i);
        }
        // reshape, permute
        if (inputSize == outputSize) {
            // reshape
            if (TensorUtils::isCopyRegion(region)) {
                return buildReshape(mLayer_, region.origin, outputs[0]);
            }
            // transpose
            if (TensorUtils::isTransposeRegion(region)) {
                return buildPermute(mLayer_, region.origin, outputs[0]);
            }
        }
        // pad
        if (inputSize < outputSize) {
            return buildPad(mLayer_, region.origin, outputs[0]);
        }
        // slice/crop
        if (inputSize > outputSize) {
            return false;
            // TODO: Apple NPU will ANCE Error.
            // return buildCrop(mLayer_, region.origin, outputs[0]);
            // return buildSlice(mLayer_, region.origin, outputs[0]);
        }
        return false;
    }
    if (TensorUtils::isDepthToSpaceRegions(outputs[0])) {
        return buildDepthToSpace(mLayer_, region.origin, outputs[0]);
    }
    // region_size > 1: concat
    {
        int dim = outputs[0]->dimensions();

        if (region.origin->dimensions() != dim) {
            return false;
        }
        int axis = -1;
        for (int i = 0; i < outputs[0]->dimensions(); i++) {
            if (region.origin->length(i) != outputs[0]->length(i)) {
                if (axis >= 0) {
                    return false;
                }
                axis = i;
            }
        }
        int elemSize = region.size[0] * region.size[1] * region.size[2];
        bool isSameShape = true;
        for (int i = 1; i < regions.size(); i++) {
            isSameShape &= (elemSize == regions[i].size[0] * regions[i].size[1] * regions[i].size[2]);
            if (regions[i].origin->dimensions() != dim) {
                return false;
            }
            for (int j = 0; j < dim; j++) {
                if (j != axis && regions[i].origin->length(j) != outputs[0]->length(j)) {
                    return false;
                }
            }
        }
        if (isSameShape && (axis - dim == -3)) {
            mCoreMLBackend->setLayerName(mLayer_, "Concat");
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONCAT;
            mLayer_->concat = mCoreMLBackend->create<CoreML__Specification__ConcatLayerParams>();
            core_ml__specification__concat_layer_params__init(mLayer_->concat);
            mLayer_->concat->sequenceconcat = false;
        } else {
            mCoreMLBackend->setLayerName(mLayer_, "NDConcat");
            mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONCAT_ND;
            mLayer_->concatnd = mCoreMLBackend->create<CoreML__Specification__ConcatNDLayerParams>();
            core_ml__specification__concat_ndlayer_params__init(mLayer_->concatnd);
            mLayer_->concatnd->axis = axis - dim;
        }
        std::vector<std::string> inputNames;
        for (const auto& reg : regions) {
            inputNames.push_back(mCoreMLBackend->getTensorName(reg.origin));
        }
        mCoreMLBackend->setLayerInputs(mLayer_, std::move(inputNames));
        return true;
    }
    return false;
}
static void dumpRegion(const Tensor::InsideDescribe::Region& reg) {
    printf("\n{\nsize: [%d, %d, %d], origin: %p\n", reg.size[0], reg.size[1], reg.size[2], reg.origin);
    printf("src: { stride: [%d, %d, %d], offset: %d }\n", reg.src.stride[0],reg.src.stride[1],reg.src.stride[2],reg.src.offset);
    printf("dst: { stride: [%d, %d, %d], offset: %d }\n}\n", reg.dst.stride[0],reg.dst.stride[1],reg.dst.stride[2],reg.dst.offset);
}
ErrorCode CoreMLRaster::onResize(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    MNN_ASSERT(outputs.size() == 1);
    if (!rasterOptimization(outputs, outputs)) {
        /*
        printf(">>> start\n");
        for (const auto& reg : TensorUtils::getDescribe(inputs[0])->regions) {
            printf("inputShape: ["); for (auto x : reg.origin->shape()) printf("%d, ", x); printf("]\n");
            dumpRegion(reg);
        }
        printf("outputShape: ["); for (auto x : outputs[0]->shape()) printf("%d, ", x); printf("]\n");
        printf(">>> end\n");*/
        auto outputShape = outputs[0]->shape();
        mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CUSTOM;
        mLayer_->custom = mCoreMLBackend->create<CoreML__Specification__CustomLayerParams>();
        core_ml__specification__custom_layer_params__init(mLayer_->custom);
        mCoreMLBackend->copyName(&(mLayer_->custom->classname), "RasterLayer");
        const auto& regions = TensorUtils::getDescribe(outputs[0])->regions;
        mLayer_->custom->n_weights = regions.size() + 1;
        mLayer_->custom->weights = mCoreMLBackend->create<CoreML__Specification__WeightParams*>(mLayer_->custom->n_weights);
        std::vector<std::string> inputNames;
        for (int i = 0; i <= regions.size(); i++) {
            mLayer_->custom->weights[i] = mCoreMLBackend->create<CoreML__Specification__WeightParams>();
            core_ml__specification__weight_params__init(mLayer_->custom->weights[i]);
            if (i == 0) {
                // first set outputShape
                mLayer_->custom->weights[i]->n_floatvalue = outputShape.size();
                mLayer_->custom->weights[i]->floatvalue = mCoreMLBackend->create<float>(mLayer_->custom->weights[i]->n_floatvalue);
                memcpy(mLayer_->custom->weights[i]->floatvalue, outputShape.data(), outputShape.size() * sizeof(int));
            } else {
                // then set regions info
                mLayer_->custom->weights[i]->n_floatvalue = 11;
                mLayer_->custom->weights[i]->floatvalue = mCoreMLBackend->create<float>(mLayer_->custom->weights[i]->n_floatvalue);
                memcpy(mLayer_->custom->weights[i]->floatvalue, &(regions[i-1]), 11 * sizeof(int));
                inputNames.push_back(mCoreMLBackend->getTensorName(regions[i-1].origin));
            }
        }
        mCoreMLBackend->setLayerInputs(mLayer_, std::move(inputNames));
    }
    mCoreMLBackend->setLayerOutputs(mLayer_, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLRaster, OpType_Raster)
} // namespace MNN
