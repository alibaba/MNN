package com.taobao.meta.avatar.a2bs

class AudioToBlendShapeData {
    var expr: List<FloatArray> = ArrayList()
    var pose: List<FloatArray> = ArrayList()
    var pose_z: List<FloatArray> = ArrayList()
    var app_pose_z: List<FloatArray> = ArrayList()
    var joints_transform: List<FloatArray> = ArrayList()
    var jaw_pose: List<FloatArray> = ArrayList()
    var frame_num: Int = 0
}