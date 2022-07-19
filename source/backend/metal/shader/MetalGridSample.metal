struct grid_sample_params {
    int batches;
    int channels;
    int inH;
    int inW;
    int outH;
    int outW;
    int mode; // 0-Bilinear, 1-Nearest
    int paddingMode; // 0-Zeros, 1-Border, 2-Reflection
    int alignCorners;
};

static float getPosition(float x, int range, int alignCorners, int paddingMode) {
    if (paddingMode == 2/*GridSamplePaddingMode_REFLECTION*/) {
        // if x is on the left side of -1.0, move it to the right side of 1.0
        if (x < -1.0f) {
            x = x + ::ceil(1 - x) * 4;
        }
        // reflect
        if (x > 1.0f) {
            float l = x - 1.0f;
            int reflectionNum = ::floor(l / 2.0);
            float offset = l - reflectionNum * 2.0f;
            x = (reflectionNum % 2 == 0) ? (1 - offset) : (-1.0f + offset);
        }
    }

    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1 + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

static ftype4 sample(int h, int w, const device ftype4 *buffer, int height, int width, int paddingMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if (paddingMode == 0/*GridSamplePaddingMode_ZEROS*/) {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
    }

    return buffer[h * width + w];
}

static ftype4 interpolate(float h, float w, const device ftype4 *buffer, int height, int width, int mode,
                         int paddingMode) {
    if (mode == 1/*GridSampleMode_NEAREST*/) {
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        return sample(nh, nw, buffer, height, width, paddingMode);
    }

    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = w0_h + 1;
    int w1_w = w0_w + 1;
    ftype4 oneV = (ftype4)((ftype)1.0f);

    ftype4 i00 = sample(w0_h, w0_w, buffer, height, width, paddingMode);
    ftype4 i01 = sample(w0_h, w1_w, buffer, height, width, paddingMode);
    ftype4 i10 = sample(w1_h, w0_w, buffer, height, width, paddingMode);
    ftype4 i11 = sample(w1_h, w1_w, buffer, height, width, paddingMode);

    
    ftype4 f0 = (ftype4)((ftype)(w1_w - w));
    ftype4 f1 = oneV - f0;
    ftype4 h0 = (ftype4)((ftype)(w1_h - h));
    ftype4 h1 = oneV - h0;

    ftype4 i0 = i00 * f0 + i01 * f1;
    ftype4 i1 = i10 * f0 + i11 * f1;

    return i0 * h0 + i1 * h1;
}

kernel void grid_sample(const device ftype4 *input   [[buffer(0)]],
                   const device ftype *grid         [[buffer(1)]],
                   device ftype4 *output             [[buffer(2)]],
                   constant grid_sample_params &p   [[buffer(3)]],
                   uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= p.outW || (int)gid.y >= p.outH || (int)gid.z >= p.batches)
        return;

    int gridPos = gid.z*p.outH*p.outW*2 + gid.y*p.outW*2 + gid.x*2;
    auto x = getPosition(grid[gridPos+0], p.inW, p.alignCorners, p.paddingMode);
    auto y = getPosition(grid[gridPos+1], p.inH, p.alignCorners, p.paddingMode);
    
    const int channelC4 = (p.channels + 3) / 4;
    for (int c = 0; c < channelC4; ++ c) {
        auto outputPos = gid.z*channelC4*p.outH*p.outW + c*p.outH*p.outW + gid.y*p.outW + gid.x;
        auto inputPtr = input + gid.z*channelC4*p.inH*p.inW + c*p.inH*p.inW;
        output[outputPos] = interpolate(y, x, inputPtr, p.inH, p.inW, p.mode, p.paddingMode);
    }
}
