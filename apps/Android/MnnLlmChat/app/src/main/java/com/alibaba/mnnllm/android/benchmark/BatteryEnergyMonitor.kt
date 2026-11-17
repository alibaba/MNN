package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.SystemClock
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

enum class EnergyUnavailableReason {
    CHARGING, UNSUPPORTED, INVALID_VOLTAGE, INVALID_CURRENT, TOO_SHORT, NO_CURRENT, INTERRUPTED,
    BELOW_BASELINE, INCOMPLETE
}

internal enum class BatteryCurrentConvention(val microampsPerUnit: Long, val dischargeSign: Long) {
    ANDROID_MICROAMPS(1, -1), OPLUS_MILLIAMPS(1000, 1);

    fun normalize(raw: Long?): Long? = raw?.let { -it * dischargeSign * microampsPerUnit }

    companion object {
        fun forDevice(manufacturer: String, model: String): BatteryCurrentConvention {
            // PJZ110 exposes the Oplus gauge's mA value with positive discharge.
            return if (manufacturer.equals("OnePlus", ignoreCase = true) && model == "PJZ110") {
                OPLUS_MILLIAMPS
            } else ANDROID_MICROAMPS
        }
    }
}

data class EnergyMeasurement(
    val joules: Double?,
    val seconds: Double,
    val reason: EnergyUnavailableReason? = null,
    val observedUpdates: Int = 0,
    val lowResolution: Boolean = false
) {
    internal var intervals: List<EnergyInterval> = emptyList()
    val averageWatts: Double? get() = joules?.takeIf { seconds > 0 }?.div(seconds)
}

internal data class BatteryEnergyReading(
    val timeNs: Long,
    val discharging: Boolean,
    val currentUa: Long?,
    val voltageMv: Int,
    val currentRaw: Long? = currentUa,
    val temperatureTenthsC: Int? = null,
    val levelPercent: Int? = null
)

internal class BatteryEnergyAccumulator {
    private var first: BatteryEnergyReading? = null
    private var previous: BatteryEnergyReading? = null
    private var error: EnergyUnavailableReason? = null
    private var energy = 0.0
    private var updates = 0
    private var samples = 0
    private var lastUpdateNs = 0L
    private var longestUnchangedSeconds = 0.0

    fun add(reading: BatteryEnergyReading) {
        if (!reading.discharging && error == null) error = EnergyUnavailableReason.CHARGING
        if (first == null) {
            first = reading
            lastUpdateNs = reading.timeNs
        }
        if (reading.voltageMv !in 2000..20000 && error == null) error = EnergyUnavailableReason.INVALID_VOLTAGE
        val current = reading.currentUa
        if (current == null && error == null) error = EnergyUnavailableReason.UNSUPPORTED
        if (current != null && current > 0 && error == null) error = EnergyUnavailableReason.INVALID_CURRENT
        previous?.let { old ->
            val seconds = (reading.timeNs - old.timeNs) / 1e9
            if (seconds <= 0 && error == null) error = EnergyUnavailableReason.TOO_SHORT
            if (old.currentUa != null && current != null) {
                energy += -(old.currentUa.toDouble() * old.voltageMv + current.toDouble() * reading.voltageMv) * 0.5e-9 * seconds
                val unchanged = (reading.timeNs - lastUpdateNs) / 1e9
                longestUnchangedSeconds = maxOf(longestUnchangedSeconds, unchanged)
                if (old.currentUa != current || old.voltageMv != reading.voltageMv) {
                    updates++
                    lastUpdateNs = reading.timeNs
                }
            }
        }
        previous = reading
        samples++
    }

    fun result(): EnergyMeasurement {
        val seconds = ((previous?.timeNs ?: 0) - (first?.timeNs ?: 0)) / 1e9
        val reason = error ?: when {
            first == null -> EnergyUnavailableReason.UNSUPPORTED
            samples < 2 || seconds <= 0 -> EnergyUnavailableReason.TOO_SHORT
            energy <= 0 -> EnergyUnavailableReason.NO_CURRENT
            else -> null
        }
        return EnergyMeasurement(energy.takeIf { reason == null }, seconds, reason,
            updates, longestUnchangedSeconds >= 1.0)
    }
}

class BatteryEnergyMonitor(context: Context) : Closeable {
    private val appContext = context.applicationContext
    private val battery = appContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    internal val currentConvention = BatteryCurrentConvention.forDevice(Build.MANUFACTURER, Build.MODEL)
    private val readings = mutableListOf<BatteryEnergyReading>()
    private var executor: java.util.concurrent.ScheduledExecutorService? = null
    private var started = false
    private var closed = false

    @Synchronized
    fun start() {
        if (started || closed) return
        started = true
        sample()
        executor = Executors.newSingleThreadScheduledExecutor { task ->
            Thread(task, "MNN-battery-current").apply { isDaemon = true }
        }.also {
            it.scheduleWithFixedDelay({ synchronized(this) { if (!closed) sample() } }, 50, 50, TimeUnit.MILLISECONDS)
        }
    }

    private fun sample() {
        val intent = try {
            appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        } catch (_: RuntimeException) { null }
        val discharging = intent != null &&
            intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1) == BatteryManager.BATTERY_STATUS_DISCHARGING &&
            intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) == 0
        val raw = try {
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                .takeUnless { it == Int.MIN_VALUE }?.toLong()
        } catch (_: RuntimeException) { null }
        val now = SystemClock.elapsedRealtimeNanos()
        val reading = BatteryEnergyReading(
            now, discharging, currentConvention.normalize(raw),
            intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0, raw,
            intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)?.takeIf { it >= 0 },
            intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)?.takeIf { it >= 0 }
        )
        readings.add(reading)
    }

    @Synchronized
    internal fun capture(): BatteryEnergyReading {
        check(started && !closed)
        sample()
        return readings.last()
    }

    @Synchronized
    internal fun snapshot(): List<BatteryEnergyReading> = readings.toList()

    @Synchronized
    internal fun measure(intervals: List<EnergyInterval>): EnergyMeasurement =
        integrateIntervals(readings, intervals)

    @Synchronized
    override fun close() {
        closed = true
        executor?.shutdownNow()
        executor = null
    }
}

internal fun integrateIntervals(
    readings: List<BatteryEnergyReading>, intervals: List<EnergyInterval>
): EnergyMeasurement {
    val seconds = intervals.sumOf { it.seconds }
    fun unavailable() = EnergyMeasurement(null, seconds, EnergyUnavailableReason.TOO_SHORT)
    if (intervals.isEmpty() || intervals.any { it.seconds <= 0 } ||
        intervals.zipWithNext().any { (a, b) -> a.endNs > b.startNs }) return unavailable()
    val parts = intervals.map { interval ->
        // Both boundaries are captured synchronously by the existing benchmark callback.
        val first = readings.binarySearchBy(interval.startNs) { it.timeNs }
        val last = readings.binarySearchBy(interval.endNs) { it.timeNs }
        if (first < 0 || last <= first) return unavailable()
        val accumulator = BatteryEnergyAccumulator()
        for (index in first..last) accumulator.add(readings[index])
        accumulator.result()
    }
    val reason = parts.firstNotNullOfOrNull { it.reason }
    return EnergyMeasurement(if (reason == null) parts.sumOf { it.joules!! } else null, seconds,
        reason, parts.sumOf { it.observedUpdates }, parts.any { it.lowResolution }).also { it.intervals = intervals }
}
