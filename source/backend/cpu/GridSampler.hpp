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
static void MNNGridSampleInterpGrad(FLOAT* outputPtr, FLOAT* inputPtr, const float* cordPtr, size_t inH, size_t inW, size_t outW, size_t channelCUnit, size_t inOffset, size_t outOffset, bool sampleMode, bool padMode) {
    const int pack = PACK;
    for (auto ow = 0; ow < outW; ++ow) {
        auto w = cordPtr[2 * ow + 0];
        auto h = cordPtr[2 * ow + 1];
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
