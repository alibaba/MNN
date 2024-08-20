struct layernorm_constants {
    int inside;
    int outside;
    float eps;
    int has_gamma_beta;
};

kernel void layernorm_x1(const device ftype *in       [[buffer(0)]],
                         device ftype *out            [[buffer(1)]],
                         constant layernorm_constants& cst  [[buffer(2)]],
                         const device float *gamma    [[buffer(3)]],
                         const device float *beta     [[buffer(4)]],
                         uint2 gid                         [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.inside || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside;
    auto out_data = out + gid.y * cst.inside;

    float mean;
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    for(int i = 0; i < cst.inside; i++) {
        sum += in_data[i];
    }
    mean = sum / cst.inside;
    
    for(int i = 0; i < cst.inside; i++) {
        float dis = (in_data[i] - mean);
        square_sum += dis * dis;
    }
    float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
    
    float norm = var * ((float)in_data[gid.x] - mean);
    if(cst.has_gamma_beta) {
        out_data[gid.x] = (ftype)(norm * gamma[gid.x] + beta[gid.x]);
    } else {
        out_data[gid.x] = (ftype)(norm);
    }
}


kernel void layernorm_x4(const device ftype4 *in       [[buffer(0)]],
                         device ftype4 *out            [[buffer(1)]],
                         constant layernorm_constants& cst  [[buffer(2)]],
                         const device float4 *gamma    [[buffer(3)]],
                         const device float4 *beta     [[buffer(4)]],
                         uint2 gid                         [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.inside/4 || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside/4;
    auto out_data = out + gid.y * cst.inside/4;

    float mean;
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    for(int i = 0; i < cst.inside/4; i++) {
        sum += in_data[i].x;
        sum += in_data[i].y;
        sum += in_data[i].z;
        sum += in_data[i].w;
    }
    mean = sum / cst.inside;
    
    for(int i = 0; i < cst.inside/4; i++) {
        float dis = (in_data[i].x - mean);
        square_sum += dis * dis;
        dis = (in_data[i].y - mean);
        square_sum += dis * dis;
        dis = (in_data[i].z - mean);
        square_sum += dis * dis;
        dis = (in_data[i].w - mean);
        square_sum += dis * dis;
    }
    float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
    
    float4 norm = var * ((float4)in_data[gid.x] - mean);
    if(cst.has_gamma_beta) {
        out_data[gid.x] = (ftype4)(norm * gamma[gid.x] + beta[gid.x]);
    } else {
        out_data[gid.x] = (ftype4)(norm);
    }
}


kernel void layernorm_x1_rms(const device ftype *in       [[buffer(0)]],
                            device ftype *out            [[buffer(1)]],
                            constant layernorm_constants& cst  [[buffer(2)]],
                            const device float *gamma    [[buffer(3)]],
                            const device float *beta     [[buffer(4)]],
                            uint2 gid                         [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.inside || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside;
    auto out_data = out + gid.y * cst.inside;

    float square_sum = 0.0f;
    
    for(int i = 0; i < cst.inside; i++) {
        float dis = in_data[i];
        square_sum += dis * dis;
    }
    float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
    
    float norm = var * ((float)in_data[gid.x]);
    if(cst.has_gamma_beta) {
        out_data[gid.x] = (ftype)(norm * gamma[gid.x] + beta[gid.x]);
    } else {
        out_data[gid.x] = (ftype)(norm);
    }
}

kernel void layernorm_x4_rms(const device ftype4 *in       [[buffer(0)]],
                             device ftype4 *out            [[buffer(1)]],
                             constant layernorm_constants& cst  [[buffer(2)]],
                             const device float4 *gamma    [[buffer(3)]],
                             const device float4 *beta     [[buffer(4)]],
                             uint2 gid                         [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.inside/4 || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside/4;
    auto out_data = out + gid.y * cst.inside/4;

    float square_sum = 0.0f;

    for(int i = 0; i < cst.inside/4; i++) {
        float dis = in_data[i].x;
        square_sum += dis * dis;
        dis = in_data[i].y;
        square_sum += dis * dis;
        dis = in_data[i].z;
        square_sum += dis * dis;
        dis = in_data[i].w;
        square_sum += dis * dis;
    }
    float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
    
    float4 norm = var * ((float4)in_data[gid.x]);
    if(cst.has_gamma_beta) {
        out_data[gid.x] = (ftype4)(norm * gamma[gid.x] + beta[gid.x]);
    } else {
        out_data[gid.x] = (ftype4)(norm);
    }
}
