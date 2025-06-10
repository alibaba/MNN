package com.taobao.meta.avatar.camera

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class Camera @JvmOverloads constructor(
    private var position: FloatArray = floatArrayOf(
        0.0f,
        0.0f,
        0.0f
    ),
    up: FloatArray = floatArrayOf(0.0f, 1.0f, 0.0f),
    var yaw: Float = YAW,
    var pitch: Float = PITCH,
    near: Float = 0.1f,
    far: Float = 100.0f
) {
    enum class CameraMovement {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT,
        UP,
        DOWN
    }

    private var front: FloatArray
    private lateinit var up: FloatArray
    private lateinit var right: FloatArray
    private var worldUp: FloatArray = up
    private var movementSpeed: Float
    private var mouseSensitivity: Float
    private var zoom: Float
    private var nearPlane: Float = near
    private var farPlane: Float = far

    init {
        front = floatArrayOf(0.0f, 0.0f, -1.0f)
        movementSpeed = SPEED
        mouseSensitivity = SENSITIVITY
        zoom = ZOOM
        updateCameraVectors()
    }

    private fun updateCameraVectors() {
        val front = FloatArray(3)
        front[0] =
            (cos(Math.toRadians(yaw.toDouble())) * cos(Math.toRadians(pitch.toDouble()))).toFloat()
        front[1] = sin(Math.toRadians(pitch.toDouble())).toFloat()
        front[2] =
            (sin(Math.toRadians(yaw.toDouble())) * cos(Math.toRadians(pitch.toDouble()))).toFloat()

        this.front = normalize(front)

        right = normalize(cross(this.front, worldUp))
        up = normalize(cross(right, this.front))
    }

    private fun cross(a: FloatArray, b: FloatArray): FloatArray {
        return floatArrayOf(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )
    }

    private fun normalize(v: FloatArray): FloatArray {
        val length = sqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).toDouble()).toFloat()
        return floatArrayOf(v[0] / length, v[1] / length, v[2] / length)
    }

    companion object {
        // Default camera values
        const val YAW: Float = -90.0f
        const val PITCH: Float = -5f
        const val SPEED: Float = 0.0f
        const val SENSITIVITY: Float = 0.1f
        const val ZOOM: Float = 50.0f // fovy
    }
}