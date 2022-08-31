#define CONV_UNROLL (4)

#define CONV_MUL_PACK_W2(x,y)  \
    x += FLOAT4(in00 * k00);\
    y += FLOAT4(in01 * k00);\
    x += FLOAT4(in01 * k01);\
    y += FLOAT4(in02 * k01);\
    x += FLOAT4(in02 * k02);\
    y += FLOAT4(in03 * k02);\
                            \
    x += FLOAT4(in10 * k10);\
    y += FLOAT4(in11 * k10);\
    x += FLOAT4(in11 * k11);\
    y += FLOAT4(in12 * k11);\
    x += FLOAT4(in12 * k12);\
    y += FLOAT4(in13 * k12);\
                            \
    x += FLOAT4(in20 * k20);\
    y += FLOAT4(in21 * k20);\
    x += FLOAT4(in21 * k21);\
    y += FLOAT4(in22 * k21);\
    x += FLOAT4(in22 * k22);\
    y += FLOAT4(in23 * k22);
                     

#define CONV_NEXT_FLT  \
    z_wt += ws;             \
                            \
    k00 = z_wt[0], k01 = z_wt[1], k02 = z_wt[2];\
    k10 = z_wt[3], k11 = z_wt[4], k12 = z_wt[5];\
    k20 = z_wt[6], k21 = z_wt[7], k22 = z_wt[8];


#define CONV_MUL_PACK_H2(x,y)  \
    x += FLOAT4(in10 * k00);\
    y += FLOAT4(in11 * k00);\
    x += FLOAT4(in11 * k01);\
    y += FLOAT4(in12 * k01);\
    x += FLOAT4(in12 * k02);\
    y += FLOAT4(in13 * k02);\
                            \
    x += FLOAT4(in20 * k10);\
    y += FLOAT4(in21 * k10);\
    x += FLOAT4(in21 * k11);\
    y += FLOAT4(in22 * k11);\
    x += FLOAT4(in22 * k12);\
    y += FLOAT4(in23 * k12);\
                            \
    x += FLOAT4(in30 * k20);\
    y += FLOAT4(in31 * k20);\
    x += FLOAT4(in31 * k21);\
    y += FLOAT4(in32 * k21);\
    x += FLOAT4(in32 * k22);\
    y += FLOAT4(in33 * k22);

struct conv_constants {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    int oz_size;
    int threadgroup_input_slice;
    
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    conv_activation_type activation;    
};

kernel void conv(const device ftype4 *in        [[buffer(0)]],
                 device ftype4 *out             [[buffer(1)]],
                 constant conv_constants& cst   [[buffer(2)]],
                 const device ftype4x4 *wt      [[buffer(3)]],
                 const device ftype4 *biasTerms [[buffer(4)]],
                 uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.oz_size) return;
    
    int idx_w = gid.x;
    int idx_h = gid.y;
    int idx_c = gid.z / cst.batch;
    int idx_b = gid.z % cst.batch;
    
    int offset_x = (int)idx_w * cst.stride_x - cst.pad_x;
    int offset_y = (int)idx_h * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in + idx_b * cst.input_slice * cst.input_size + offset_y * cst.input_width + offset_x;
    auto z_wt  = wt + idx_c * cst.input_slice * cst.kernel_size + sy * cst.kernel_x + sx;
    auto z_out = out + idx_b * cst.output_slice * cst.output_size + (int)idx_c * cst.output_size                   + (int)gid.y * cst.output_width + (int)gid.x;

    int dilation_h = cst.input_width * cst.dilation_y;
    FLOAT4 result = FLOAT4(biasTerms[idx_c]);
    for (auto z = 0; z < cst.input_slice; z++) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto wt4 = z_wt[z * cst.kernel_size + y * cst.kernel_x + x];
                auto in4 = z_in[z * cst.input_size  + y * dilation_h   + x * cst.dilation_x];
                result += FLOAT4(in4 * wt4);
            }
        }
    }
    *z_out = activate(ftype4(result), cst.activation);
}

kernel void convk3s1d1p1_w2z4(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int idx_w = gid.x << 1;
    int idx_h = gid.y;
    int idx_c = gid.z / cst.batch;
    int idx_b = gid.z % cst.batch;

    int4 uz = idx_c * CONV_UNROLL + int4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;

    bool valid_x = (int)(gid.x * 2 + 1) < cst.output_width;

    int offset_x = (int)gid.x * 2 - cst.pad_x;
    int offset_y = (int)gid.y - cst.pad_y;

    auto z_in  = in + idx_b * cst.input_slice * cst.input_size + offset_y * cst.input_width + offset_x;
    auto z_flt  = wt  + uz[0] * cst.input_slice * cst.kernel_size;
    auto z_out = out + idx_b * cst.output_slice * cst.output_size + uz[0] * cst.output_size + idx_h * cst.output_width + idx_w;
    
    int ws = cst.input_slice * cst.kernel_size;
    FLOAT4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    FLOAT4 result4 = 0, result5 = 0, result6 = 0, result7 = 0;

    for (auto z = 0; z < cst.input_slice; z++, z_flt += cst.kernel_size, z_in += cst.input_size) {
        auto in00 = (offset_x<0                   || offset_y<0) ? (ftype4)0.f : *(z_in+0*cst.input_width+0);
        auto in01 = (offset_x+1>=cst.input_width  || offset_y<0) ? (ftype4)0.f : *(z_in+0*cst.input_width+1);
        auto in02 = (offset_x+2>=cst.input_width  || offset_y<0) ? (ftype4)0.f : *(z_in+0*cst.input_width+2);
        auto in03 = (offset_x+3>=cst.input_width  || offset_y<0) ? (ftype4)0.f : *(z_in+0*cst.input_width+3);

        auto in10 = (offset_x<0                   || offset_y+1>=cst.input_height) ? (ftype4)0.f : *(z_in+1*cst.input_width+0);
        auto in11 = (offset_x+1>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : *(z_in+1*cst.input_width+1);
        auto in12 = (offset_x+2>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : *(z_in+1*cst.input_width+2);
        auto in13 = (offset_x+3>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : *(z_in+1*cst.input_width+3);
        
        auto in20 = (offset_x<0                   || offset_y+2>=cst.input_height) ? (ftype4)0.f : *(z_in+2*cst.input_width+0);
        auto in21 = (offset_x+1>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : *(z_in+2*cst.input_width+1);
        auto in22 = (offset_x+2>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : *(z_in+2*cst.input_width+2);
        auto in23 = (offset_x+3>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : *(z_in+2*cst.input_width+3);
        
        auto z_wt = z_flt;
        auto k00 = z_wt[0], k01 = z_wt[1], k02 = z_wt[2];
        auto k10 = z_wt[3], k11 = z_wt[4], k12 = z_wt[5];
        auto k20 = z_wt[6], k21 = z_wt[7], k22 = z_wt[8];

        CONV_MUL_PACK_W2(result0,result4);
        if (valids[0]) {
            CONV_NEXT_FLT;
            CONV_MUL_PACK_W2(result1,result5);
        }
        if (valids[1]) {
            CONV_NEXT_FLT;
            CONV_MUL_PACK_W2(result2,result6);
        }
        if (valids[2]) {
            CONV_NEXT_FLT;
            CONV_MUL_PACK_W2(result3,result7);
        }
    }
    /* true */ *z_out = activate(ftype4(result0 + float4(biasTerms[uz[0]])), cst.activation);
    if(valid_x) {
        *(z_out+1) = activate(ftype4(result4 + float4(biasTerms[uz[0]])), cst.activation);
    }

    if (valids[0]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result1 + float4(biasTerms[uz[1]])), cst.activation);
        if(valid_x) {
            *(z_out+1) = activate(ftype4(result5 + float4(biasTerms[uz[1]])), cst.activation);
        }
    }
    if (valids[1]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result2 + float4(biasTerms[uz[2]])), cst.activation);
        if(valid_x) {
            *(z_out+1) = activate(ftype4(result6 + float4(biasTerms[uz[2]])), cst.activation);
        }
    }
    if (valids[2]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result3 + float4(biasTerms[uz[3]])), cst.activation);
        if(valid_x) {
            *(z_out+1) = activate(ftype4(result7 + float4(biasTerms[uz[3]])), cst.activation);
        }
    }
}


kernel void conv_z4(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int idx_w = gid.x;
    int idx_h = gid.y;
    int idx_c = gid.z / cst.batch;
    int idx_b = gid.z % cst.batch;
    
    int4 uz = idx_c * CONV_UNROLL + int4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;
    
    int offset_x = idx_w * cst.stride_x - cst.pad_x;
    int offset_y = idx_h * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in + idx_b * cst.input_slice * cst.input_size + offset_y * cst.input_width + offset_x;
    auto z_wt  = wt  + uz[0] * cst.input_slice * cst.kernel_size + sy * cst.kernel_x + sx;
    auto z_out = out + idx_b * cst.output_slice * cst.output_size + uz[0] * cst.output_size + idx_h * cst.output_width + idx_w;
    
    int ws = cst.input_slice * cst.kernel_size;
    int dilation_h = cst.input_width * cst.dilation_y;
    FLOAT4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < cst.input_slice; z++, z_wt += cst.kernel_size, z_in += cst.input_size) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto x_wt = z_wt + y * cst.kernel_x + x;
                auto in4  = z_in[  y * dilation_h   + x * cst.dilation_x];
                /* true                   */ result0 += FLOAT4(in4 * *x_wt);
                if (valids[0]) { x_wt += ws; result1 += FLOAT4(in4 * *x_wt); }
                if (valids[1]) { x_wt += ws; result2 += FLOAT4(in4 * *x_wt); }
                if (valids[2]) { x_wt += ws; result3 += FLOAT4(in4 * *x_wt); }
            }
        }
    }
    /* true                                 */ *z_out = activate(ftype4(result0 + FLOAT4(biasTerms[uz[0]])), cst.activation);
    if (valids[0]) { z_out += cst.output_size; *z_out = activate(ftype4(result1 + FLOAT4(biasTerms[uz[1]])), cst.activation); }
    if (valids[1]) { z_out += cst.output_size; *z_out = activate(ftype4(result2 + FLOAT4(biasTerms[uz[2]])), cst.activation); }
    if (valids[2]) { z_out += cst.output_size; *z_out = activate(ftype4(result3 + FLOAT4(biasTerms[uz[3]])), cst.activation); }
}
