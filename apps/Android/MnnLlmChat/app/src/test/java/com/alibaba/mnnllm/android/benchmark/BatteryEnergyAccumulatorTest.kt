package com.alibaba.mnnllm.android.benchmark

import org.junit.Assert.*
import org.junit.Test

class BatteryEnergyAccumulatorTest {
    private fun reading(seconds: Double, current: Long? = -500_000, voltage: Int = 4000,
                        discharging: Boolean = true) =
        BatteryEnergyReading((seconds * 1e9).toLong(), discharging, current, voltage)

    @Test
    fun integratesMicroampsAndMillivoltsAcrossUnevenIntervals() {
        val accumulator = BatteryEnergyAccumulator()
        accumulator.add(reading(0.0, -500_000))
        accumulator.add(reading(1.0, -1_000_000))
        accumulator.add(reading(3.0, -500_000))
        val result = accumulator.result()
        assertNull(result.reason)
        assertEquals(9.0, result.joules!!, 1e-9)
        assertEquals(3.0, result.averageWatts!!, 1e-9)
    }

    @Test
    fun constantCurrentDoesNotRequireAnyCounterOrValueChange() {
        val accumulator = BatteryEnergyAccumulator()
        accumulator.add(reading(0.0))
        accumulator.add(reading(5.0))
        accumulator.add(reading(10.0))
        val result = accumulator.result()
        assertNull(result.reason)
        assertEquals(20.0, result.joules!!, 1e-9)
        assertEquals(0, result.observedUpdates)
        assertTrue(result.lowResolution)
    }

    @Test
    fun changingVoltageIsIntegratedWithTheCorrespondingCurrent() {
        val accumulator = BatteryEnergyAccumulator()
        accumulator.add(reading(0.0, -500_000, 4000))
        accumulator.add(reading(2.0, -1_000_000, 3000))
        assertEquals(5.0, accumulator.result().joules!!, 1e-9)
    }

    @Test
    fun chargingAtAnyPointInvalidatesWholeWindow() {
        val accumulator = BatteryEnergyAccumulator()
        accumulator.add(reading(0.0))
        accumulator.add(reading(1.0, discharging = false))
        accumulator.add(reading(2.0))
        assertEquals(EnergyUnavailableReason.CHARGING, accumulator.result().reason)
        assertNull(accumulator.result().joules)
    }

    @Test
    fun missingOrWrongDirectionCurrentIsNotReplacedByAnotherMeasurement() {
        val missing = BatteryEnergyAccumulator()
        missing.add(reading(0.0, null))
        missing.add(reading(2.0, null))
        assertEquals(EnergyUnavailableReason.UNSUPPORTED, missing.result().reason)
        val reversed = BatteryEnergyAccumulator()
        reversed.add(reading(0.0, 500_000))
        reversed.add(reading(2.0, 500_000))
        assertEquals(EnergyUnavailableReason.INVALID_CURRENT, reversed.result().reason)
        assertNull(reversed.result().joules)
    }

    @Test
    fun zeroCurrentDoesNotBecomeAZeroPowerClaim() {
        val accumulator = BatteryEnergyAccumulator()
        accumulator.add(reading(0.0, 0))
        accumulator.add(reading(10.0, 0))
        assertEquals(EnergyUnavailableReason.NO_CURRENT, accumulator.result().reason)
        assertNull(accumulator.result().averageWatts)
    }

    @Test
    fun requiresValidVoltageAndMonotonicTime() {
        val voltage = BatteryEnergyAccumulator()
        voltage.add(reading(0.0, voltage = 0))
        voltage.add(reading(1.0, voltage = 0))
        assertEquals(EnergyUnavailableReason.INVALID_VOLTAGE, voltage.result().reason)
        val time = BatteryEnergyAccumulator()
        time.add(reading(1.0))
        time.add(reading(1.0))
        assertEquals(EnergyUnavailableReason.TOO_SHORT, time.result().reason)
    }

    @Test
    fun knownOplusConventionAndStandardAndroidProduceTheSamePhysicalCurrent() {
        assertEquals(-500_000L, BatteryCurrentConvention.ANDROID_MICROAMPS.normalize(-500_000L))
        assertEquals(-500_000L, BatteryCurrentConvention.OPLUS_MILLIAMPS.normalize(500L))
        assertEquals(BatteryCurrentConvention.OPLUS_MILLIAMPS,
            BatteryCurrentConvention.forDevice("OnePlus", "PJZ110"))
        assertEquals(BatteryCurrentConvention.ANDROID_MICROAMPS,
            BatteryCurrentConvention.forDevice("OnePlus", "unknown"))
        assertNull(BatteryCurrentConvention.ANDROID_MICROAMPS.normalize(null))
    }
}
