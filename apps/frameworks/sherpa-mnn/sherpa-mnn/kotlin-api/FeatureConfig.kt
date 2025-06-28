package com.k2fsa.sherpa.mnn

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
)

fun getFeatureConfig(sampleRate: Int, featureDim: Int): FeatureConfig {
    return FeatureConfig(sampleRate = sampleRate, featureDim = featureDim)
}
