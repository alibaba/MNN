package com.alibaba.mnnllm.android.modelmarket

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelRepositoryVersionRulesTest {

    @Test
    fun normalizeVersion_shouldStripGpSuffixes() {
        assertEquals("0.8.0.1", ModelRepository.normalizeVersionForComparison("0.8.0.1-gp"))
        assertEquals("0.8.0.1", ModelRepository.normalizeVersionForComparison("0.8.0.1_gp"))
        assertEquals("0.8.0.1", ModelRepository.normalizeVersionForComparison("0.8.0.1"))
    }

    @Test
    fun compareVersions_shouldTreatSuffixVersionAsEqual() {
        assertEquals(0, ModelRepository.compareVersions("0.8.0.1-gp", "0.8.0.1"))
        assertEquals(0, ModelRepository.compareVersions("0.8.0.1_gp", "0.8.0.1"))
    }

    @Test
    fun compareVersions_shouldCompareSegmentBySegment() {
        assertTrue(ModelRepository.compareVersions("0.8.0.2", "0.8.0.1") > 0)
        assertTrue(ModelRepository.compareVersions("0.8.1", "0.8.0.9") > 0)
        assertTrue(ModelRepository.compareVersions("1", "0.9.9.9") > 0)
    }

    @Test
    fun isAppVersionSupported_shouldUseMinAppVersionRule() {
        assertTrue(ModelRepository.isAppVersionSupported("0.8.0.1-gp", "0.8.0.1"))
        assertFalse(ModelRepository.isAppVersionSupported("0.8.0.1_gp", "0.8.0.2"))
        assertTrue(ModelRepository.isAppVersionSupported("0.8.0.1", null))
        assertTrue(ModelRepository.isAppVersionSupported("0.8.0.1", ""))
    }
}
