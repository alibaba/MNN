package com.alibaba.mnnllm.android.benchmark

data class EnergyInterval(val startNs: Long, val endNs: Long) {
    val seconds: Double get() = (endNs - startNs) / 1e9
}

internal class BenchmarkEnergySamples {
    private var startNs: Long? = null
    val completed = mutableListOf<EnergyInterval>()
    var invalid: Boolean = false
        private set
    val inFlight: Boolean get() = startNs != null

    fun record(progress: BenchmarkProgress, timeNs: Long) {
        if (progress.progressType != ProgressType.RUNNING_TEST || invalid) return
        if (progress.currentIteration != completed.size + 1) {
            invalid = true
            return
        }
        val start = startNs
        // RUNNING_TEST is sent before each round, then again with its measured duration.
        if (progress.runTimeSeconds == 0f && start == null) {
            startNs = timeNs
        } else if (progress.runTimeSeconds > 0f && start != null && timeNs > start) {
            completed.add(EnergyInterval(start, timeNs))
            startNs = null
        } else {
            invalid = true
        }
    }
}

data class BenchmarkEnergyResult(
    val inference: EnergyMeasurement,
    val baseline: EnergyMeasurement,
    val samples: List<EnergyInterval>,
    val failure: EnergyUnavailableReason? = null
) {
    val baselineWatts: Double? get() = baseline.averageWatts
    val incrementalJoules: Double? get() = inference.joules?.let { joules ->
        baselineWatts?.let { watts -> joules - watts * inference.seconds }
    }
    val incrementalWatts: Double? get() = incrementalJoules?.takeIf { inference.seconds > 0 }?.div(inference.seconds)
    val reason: EnergyUnavailableReason? get() = failure ?: inference.reason ?: baseline.reason ?: when {
        samples.isEmpty() || inference.seconds <= 0 -> EnergyUnavailableReason.INCOMPLETE
        baseline.seconds < 10.0 -> EnergyUnavailableReason.INCOMPLETE
        incrementalJoules == null -> EnergyUnavailableReason.UNSUPPORTED
        incrementalJoules!! <= 0 -> EnergyUnavailableReason.BELOW_BASELINE
        else -> null
    }
    val displaysTotalEnergy: Boolean get() = reason == EnergyUnavailableReason.BELOW_BASELINE
    val displayJoules: Double? get() = when (reason) {
        null -> incrementalJoules
        EnergyUnavailableReason.BELOW_BASELINE -> inference.joules
        else -> null
    }
    val displayWatts: Double? get() = displayJoules?.takeIf { inference.seconds > 0 }?.div(inference.seconds)
}
