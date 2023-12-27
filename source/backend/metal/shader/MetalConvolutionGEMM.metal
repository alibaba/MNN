
struct matmul4x4_const {
    int output_width;
    int output_height;
    int multi_length;
    int group;
};

template <typename IType, typename OType>
static inline void matmul4x4_template(const device IType *in,
                                      device OType *out,
                                      const device IType *kt,
                                      constant matmul4x4_const &cst,
                                      uint3 gid) {
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height) {
        auto ky = (int)gid.y + (int)gid.z * cst.output_height;
        auto iy = (int)gid.x + (int)gid.z * cst.output_width;
        auto off_in  = in  + iy * cst.multi_length;
        auto off_wt  = kt  + ky * cst.multi_length;
        auto off_out = out + iy + 4 * (int)gid.y * cst.output_width * cst.group;
        
        FLOAT4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
        for (int k = 0; k < cst.multi_length; ++k) {
            auto w4x4 = off_wt[k];
            auto i4x4 = off_in[k];
            result0 += FLOAT4(w4x4 * i4x4[0]);
            result1 += FLOAT4(w4x4 * i4x4[1]);
            result2 += FLOAT4(w4x4 * i4x4[2]);
            result3 += FLOAT4(w4x4 * i4x4[3]);
        }
        *off_out = OType(result0); off_out += cst.output_width * cst.group;
        *off_out = OType(result1); off_out += cst.output_width * cst.group;
        *off_out = OType(result2); off_out += cst.output_width * cst.group;
        *off_out = OType(result3);
    }
}

kernel void matmul4x4(const device ftype4x4 *in     [[buffer(0)]],
                      device ftype4 *out            [[buffer(1)]],
                      const device ftype4x4 *kt     [[buffer(2)]],
                      constant matmul4x4_const &cst [[buffer(3)]],
                      uint3 gid                     [[thread_position_in_grid]]) {
    matmul4x4_template<ftype4x4, ftype4>(in, out, kt, cst, gid);
}
