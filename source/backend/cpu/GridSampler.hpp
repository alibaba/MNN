static int MNNGridSampleComputeOffset(int h, int w, int height, int width, bool padMode) {
    if (padMode == true) { //padMode == BorderMode_ZEROS
        if (h < 0 || h >= height || w < 0 || w >= width) {
            return -1;
        }
    } else {
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = h < 0 ? 0 : (h > (height - 1) ? (height - 1) : h);
        w = w < 0 ? 0 : (w > (width - 1) ? (width - 1) : w);
    }
    return h * width * PACK + w * PACK;
}

static void MNNGridSampleInterp(FLOAT* outputPtr, const FLOAT* inputPtr, const FLOAT* cordPtr, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode) {
    for (auto ow = 0; ow < outW; ++ow) {
        auto w_ = cordPtr[2 * ow + 0];
        auto h_ = cordPtr[2 * ow + 1];
        float w = (float)(w_);
        float h = (float)(h_);
        Vec interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            int ns = MNNGridSampleComputeOffset(nh, nw, inH, inW, padMode);
            for (int k = 0; k < channelCUnit; ++k) {
                interp = ns == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + ns);
                Vec::save(outputPtr + k * outOffset + PACK * ow, interp);
            }
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = Vec(1.0f);

            auto f0 = Vec((FLOAT)w1_w - w_);
            auto f1 = oneV - f0;
            auto h0 = Vec((FLOAT)w1_h - h_);
            auto h1 = oneV - h0;

            int s00 = MNNGridSampleComputeOffset(w0_h, w0_w, inH, inW, padMode);
            int s01 = MNNGridSampleComputeOffset(w0_h, w1_w, inH, inW, padMode);
            int s10 = MNNGridSampleComputeOffset(w1_h, w0_w, inH, inW, padMode);
            int s11 = MNNGridSampleComputeOffset(w1_h, w1_w, inH, inW, padMode);

            for (int k = 0; k < channelCUnit; ++k) {
                Vec i00 = s00 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s00);
                Vec i01 = s01 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s01);
                Vec i10 = s10 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s10);
                Vec i11 = s11 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s11);

                Vec i0 = i00 * f0 + i01 * f1;
                Vec i1 = i10 * f0 + i11 * f1;

                interp = i0 * h0 + i1 * h1;
                Vec::save(outputPtr + k * outOffset + PACK * ow, interp);
            }
        }
    }
}
static void MNNGridSampleInterpGrad(FLOAT* outputPtr, FLOAT* inputPtr, const FLOAT* cordPtr, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode) {
    const int pack = PACK;
    for (auto ow = 0; ow < outW; ++ow) {
        auto w_ = cordPtr[2 * ow + 0];
        auto h_ = cordPtr[2 * ow + 1];
        float w = (float)(w_);
        float h = (float)(h_);
        Vec interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            int ns = MNNGridSampleComputeOffset(nh, nw, inH, inW, padMode);
            if (ns != -1) {
                for (int k = 0; k < channelCUnit; ++k) {
                    auto o = Vec::load(outputPtr + k * outOffset + pack * ow);
                    auto i = Vec::load(inputPtr + k * inOffset + ns);
                    Vec::save(inputPtr + k * inOffset + ns, i + o);
                }
            }
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = Vec(1.0f);

            auto f0 = Vec((float)w1_w - w);
            auto f1 = oneV - f0;
            auto h0 = Vec((float)w1_h - h);
            auto h1 = oneV - h0;

            int s00 = MNNGridSampleComputeOffset(w0_h, w0_w, inH, inW, padMode);
            int s01 = MNNGridSampleComputeOffset(w0_h, w1_w, inH, inW, padMode);
            int s10 = MNNGridSampleComputeOffset(w1_h, w0_w, inH, inW, padMode);
            int s11 = MNNGridSampleComputeOffset(w1_h, w1_w, inH, inW, padMode);

            for (int k = 0; k < channelCUnit; ++k) {
                auto o = Vec::load(outputPtr + k * outOffset + pack * ow);
                if (s00 != -1) {
                    auto i = Vec::load(inputPtr + k * inOffset + s00);
                    auto diff = o * h0 * f0;
                    Vec::save(inputPtr + k * inOffset + s00, diff + i);
                }
                if (s01 != -1) {
                    auto i = Vec::load(inputPtr + k * inOffset + s01);
                    auto diff = o * h0 * f1;
                    Vec::save(inputPtr + k * inOffset + s01, diff + i);
                }
                if (s10 != -1) {
                    auto i = Vec::load(inputPtr + k * inOffset + s10);
                    auto diff = o * h1 * f0;
                    Vec::save(inputPtr + k * inOffset + s10, diff + i);
                }
                if (s11 != -1) {
                    auto i = Vec::load(inputPtr + k * inOffset + s11);
                    auto diff = o * h1 * f1;
                    Vec::save(inputPtr + k * inOffset + s11, diff + i);
                }
            }
        }
    }
}

static int MNNGridSampleComputeOffset3D(int d, int h, int w, int depth, int height, int width, bool padMode) {
    if (padMode == true) { //padMode == BorderMode_ZEROS
        if (h < 0 || h >= height || w < 0 || w >= width || d < 0 || d >= depth) {
            return -1;
        }
    } else {
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        d = d < 0 ? 0 : (d > (depth - 1) ? (depth - 1) : d);
        h = h < 0 ? 0 : ( h > (height - 1) ? (height - 1) : h);
        w = w < 0 ? 0 : ( w > (width - 1) ? (width - 1) : w);
    }
    return ((d * height + h) * width + w) * 4;
}

static void MNNGridSampleInterp3D(FLOAT* outputPtr, const FLOAT* inputPtr, const FLOAT* cordPtr, size_t inD, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode) {
    for (auto ow = 0; ow < outW; ++ow) {
        auto w_ = cordPtr[3 * ow + 0];
        auto h_ = cordPtr[3 * ow + 1];
        auto d_ = cordPtr[3 * ow + 2];
        float w = (float)(w_);
        float h = (float)(h_);
        float d = (float)(d_);

        Vec interp;

        if (sampleMode == true) { //sampleMode == SampleMode_NEAREST
            int nd = ::floor(d + 0.5f);
            int nh = ::floor(h + 0.5f);
            int nw = ::floor(w + 0.5f);
            size_t ns = MNNGridSampleComputeOffset3D(nd, nh, nw, inD, inH, inW, padMode);
            for (int k = 0; k < channelCUnit; ++k) {
                interp = ns == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + ns);
                Vec::save(outputPtr + k * outOffset + PACK * ow, interp);
            }
        } else { //sampleMode == GridSampleMode_BILINEAR
            int w0_d = ::floor(d);
            int w0_h = ::floor(h);
            int w0_w = ::floor(w);
            int w1_d = ::ceil(d);
            int w1_h = ::ceil(h);
            int w1_w = ::ceil(w);
            auto oneV = Vec(1.0f);

            auto f0 = Vec((float)w1_w - w);
            auto f1 = oneV - f0;
            auto h0 = Vec((float)w1_h - h);
            auto h1 = oneV - h0;
            auto d0 = Vec((float)w1_d - d);
            auto d1 = oneV - d0;

            size_t s000 = MNNGridSampleComputeOffset3D(w0_d, w0_h, w0_w, inD, inH, inW, padMode);
            size_t s001 = MNNGridSampleComputeOffset3D(w0_d, w0_h, w1_w, inD, inH, inW, padMode);
            size_t s010 = MNNGridSampleComputeOffset3D(w0_d, w1_h, w0_w, inD, inH, inW, padMode);
            size_t s011 = MNNGridSampleComputeOffset3D(w0_d, w1_h, w1_w, inD, inH, inW, padMode);
            size_t s100 = MNNGridSampleComputeOffset3D(w1_d, w0_h, w0_w, inD, inH, inW, padMode);
            size_t s101 = MNNGridSampleComputeOffset3D(w1_d, w0_h, w1_w, inD, inH, inW, padMode);
            size_t s110 = MNNGridSampleComputeOffset3D(w1_d, w1_h, w0_w, inD, inH, inW, padMode);
            size_t s111 = MNNGridSampleComputeOffset3D(w1_d, w1_h, w1_w, inD, inH, inW, padMode);

            for (int k = 0; k < channelCUnit; ++k) {
                Vec i000 = s000 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s000);
                Vec i001 = s001 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s001);
                Vec i010 = s010 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s010);
                Vec i011 = s011 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s011);
                Vec i100 = s100 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s100);
                Vec i101 = s101 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s101);
                Vec i110 = s110 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s110);
                Vec i111 = s111 == -1 ? Vec(0.f) : Vec::load(inputPtr + k * inOffset + s111);

                Vec i00 = i000 * f0 + i001 * f1;
                Vec i01 = i010 * f0 + i011 * f1;
                Vec i0 = i00 * h0 + i01 * h1;
                Vec i10 = i100 * f0 + i101 * f1;
                Vec i11 = i110 * f0 + i111 * f1;
                Vec i1 = i10 * h0 + i11 * h1;
                interp = i0 * d0 + i1 * d1;

                Vec::save(outputPtr + k * outOffset + PACK * ow, interp);
            }
        }
    }
}
