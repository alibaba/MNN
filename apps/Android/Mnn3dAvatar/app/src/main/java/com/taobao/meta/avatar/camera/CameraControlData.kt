package com.taobao.meta.avatar.camera

import android.opengl.Matrix

class CameraControlData {
    @JvmField
    var lastScale: Float = 1.0f
    @JvmField
    var curScale: Float = 1.0f
    var rotateX: FloatArray = FloatArray(16)
    @JvmField
    var rotateY: FloatArray = FloatArray(16)
    @JvmField
    var scaleMatrix: FloatArray = FloatArray(16)
    @JvmField
    var modelMatrix: FloatArray = FloatArray(16)
    @JvmField
    var rotateSpeed: Float
    var isFirstFrame: Boolean
    var camera: Camera = Camera()
    @JvmField
    var distanceOnScreen: Float

    init {
        Matrix.setIdentityM(this.rotateX, 0)
        Matrix.setIdentityM(this.rotateY, 0)
        Matrix.setIdentityM(this.scaleMatrix, 0)
        Matrix.setIdentityM(this.modelMatrix, 0)

        this.rotateSpeed = 0.01f
        this.isFirstFrame = true
        this.camera = Camera(
            floatArrayOf(0.0f, 1.0f, 2.2f),
            floatArrayOf(0.0f, 1.0f, 0.0f),
            Camera.YAW,
            Camera.PITCH,
            0.1f,
            100.0f
        )
        this.distanceOnScreen = 0.0f
    }
}