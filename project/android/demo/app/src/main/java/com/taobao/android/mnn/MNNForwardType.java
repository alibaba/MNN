package com.taobao.android.mnn;

public enum MNNForwardType {
    /**
     * CPU
     */
    FORWARD_CPU(0),
    /**
     * OPENCL
     */
    FORWARD_OPENCL(3),
    /**
     * AUTO
     */
    FORWARD_AUTO(4),
    /**
     * OPENGL
     */
    FORWARD_OPENGL(6),
    /**
     * VULKAN
     */
    FORWARD_VULKAN(7),
    /**
     * NNAPI 
     */
    FORWARD_VULKAN(5),

    FORWARD_NeuronAdaptAPI(11);

    public int type;

    MNNForwardType(int t) {
        type = t;
    }
}
