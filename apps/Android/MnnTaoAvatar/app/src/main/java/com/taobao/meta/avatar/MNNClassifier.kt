package com.taobao.meta.avatar

object MNNClassifier {
    init {
        // Use the same native library loaded by app; keep this minimal.
        System.loadLibrary("taoavatar")
    }

    /**
     * Initialize the classifier with absolute model path.
     * Returns true on success, false otherwise.
     */
    external fun init(modelPath: String): Boolean

    /**
     * Run inference. Expects a FloatArray matching model input element count.
     * Returns predicted class index >= 0 on success, -1 on error.
     */
    external fun run(input: FloatArray): Int
}
