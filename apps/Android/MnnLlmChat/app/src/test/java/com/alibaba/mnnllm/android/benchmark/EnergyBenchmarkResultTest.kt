package com.alibaba.mnnllm.android.benchmark

import org.junit.Assert.*
import org.junit.Test

class EnergyBenchmarkResultTest {
    private fun samples() = listOf(EnergyInterval(0, 4_000_000_000))

    @Test
    fun subtractsOnlyPostTestBaselineOverActualInferenceTime() {
        val result = BenchmarkEnergyResult(EnergyMeasurement(20.0, 4.0),
            EnergyMeasurement(10.0, 10.0), samples())
        assertNull(result.reason)
        assertEquals(1.0, result.baselineWatts!!, 1e-9)
        assertEquals(16.0, result.incrementalJoules!!, 1e-9)
        assertEquals(4.0, result.incrementalWatts!!, 1e-9)
        assertFalse(result.displaysTotalEnergy)
        assertEquals(16.0, result.displayJoules!!, 1e-9)
    }

    @Test
    fun incompleteBaselineAndInterruptionsCannotProduceAcceptedResults() {
        val result = BenchmarkEnergyResult(EnergyMeasurement(20.0, 4.0),
            EnergyMeasurement(9.0, 9.0), samples())
        assertEquals(EnergyUnavailableReason.INCOMPLETE, result.reason)
        assertNull(result.displayJoules)
        assertEquals(EnergyUnavailableReason.INTERRUPTED,
            result.copy(failure = EnergyUnavailableReason.INTERRUPTED).reason)
    }

    @Test
    fun negativeIncreaseIsPreservedAndCannotBecomeZeroEnergy() {
        val result = BenchmarkEnergyResult(EnergyMeasurement(20.0, 4.0),
            EnergyMeasurement(60.0, 10.0), samples())
        assertEquals(-4.0, result.incrementalJoules!!, 1e-9)
        assertEquals(EnergyUnavailableReason.BELOW_BASELINE, result.reason)
        assertTrue(result.displaysTotalEnergy)
        assertEquals(20.0, result.displayJoules!!, 1e-9)
        assertEquals(5.0, result.displayWatts!!, 1e-9)
        val zeroIncrease = result.copy(baseline = EnergyMeasurement(50.0, 10.0))
        assertEquals(0.0, zeroIncrease.incrementalJoules!!, 1e-9)
        assertEquals(20.0, zeroIncrease.displayJoules!!, 1e-9)
    }

    @Test
    fun chargingBaselineInvalidatesEnergy() {
        val result = BenchmarkEnergyResult(EnergyMeasurement(20.0, 4.0),
            EnergyMeasurement(null, 10.0, EnergyUnavailableReason.CHARGING), samples())
        assertEquals(EnergyUnavailableReason.CHARGING, result.reason)
        assertFalse(result.displaysTotalEnergy)
        assertNull(result.displayJoules)
    }

    private fun progress(iteration: Int, duration: Float = 0f) = BenchmarkProgress(
        0, "", ProgressType.RUNNING_TEST, iteration, 3, 1024, 128, duration)

    @Test
    fun existingCallbacksExcludeWarmupAndIgnoreLegacyLogCallbacks() {
        val samples = BenchmarkEnergySamples()
        samples.record(progress(0).copy(progressType = ProgressType.WARMING_UP), 0)
        samples.record(progress(1), 1_000_000_000)
        samples.record(progress(1, 1f), 2_000_000_000)
        samples.record(BenchmarkProgress(-1, "legacy iteration log"), 3_000_000_000)
        samples.record(progress(2), 4_000_000_000)
        samples.record(progress(2, 1f), 5_000_000_000)
        assertFalse(samples.invalid)
        assertFalse(samples.inFlight)
        assertEquals(listOf(EnergyInterval(1_000_000_000, 2_000_000_000),
            EnergyInterval(4_000_000_000, 5_000_000_000)), samples.completed)
    }

    @Test
    fun unfinishedOrMismatchedCallbacksAreNotAccepted() {
        val unfinished = BenchmarkEnergySamples()
        unfinished.record(progress(1), 1_000_000_000)
        assertTrue(unfinished.inFlight)
        assertTrue(unfinished.completed.isEmpty())
        unfinished.record(progress(2, 1f), 2_000_000_000)
        assertTrue(unfinished.invalid)
        assertTrue(unfinished.completed.isEmpty())
        val missingStart = BenchmarkEnergySamples()
        missingStart.record(progress(1, 1f), 1_000_000_000)
        assertTrue(missingStart.invalid)
    }

    @Test
    fun excludesWarmupAndGapsFromInferenceEnergy() {
        val readings = listOf(
            BatteryEnergyReading(0, true, -2_500_000, 4000),
            BatteryEnergyReading(1_000_000_000, true, -750_000, 4000),
            BatteryEnergyReading(2_000_000_000, true, -750_000, 4000),
            BatteryEnergyReading(3_000_000_000, true, -2_500_000, 4000),
            BatteryEnergyReading(4_000_000_000, true, -750_000, 4000),
            BatteryEnergyReading(5_000_000_000, true, -750_000, 4000),
            BatteryEnergyReading(6_000_000_000, true, -250_000, 4000))
        val measured = integrateIntervals(readings,
            listOf(EnergyInterval(1_000_000_000, 2_000_000_000), EnergyInterval(4_000_000_000, 5_000_000_000)))
        assertNull(measured.reason)
        assertEquals(6.0, measured.joules!!, 1e-9)
        assertEquals(2.0, measured.seconds, 1e-9)
        assertEquals(EnergyUnavailableReason.TOO_SHORT,
            integrateIntervals(readings, listOf(EnergyInterval(0, 7_000_000_000))).reason)
    }
}
