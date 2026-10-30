package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.LinearLayout
import androidx.core.view.ViewCompat
import androidx.core.view.isVisible
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.ViewBenchmarkEnergyDetailRowBinding
import com.alibaba.mnnllm.android.databinding.ViewBenchmarkEnergyResultBinding

class BenchmarkEnergyResultView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {
    private val binding = ViewBenchmarkEnergyResultBinding.inflate(LayoutInflater.from(context), this, true)

    init {
        orientation = VERTICAL
        binding.energyDetailsToggle.setOnClickListener {
            setDetailsExpanded(!binding.energyDetails.isVisible)
        }
    }

    fun bind(test: TestInstance) {
        val result = requireNotNull(test.energy)
        val reason = result.reason
        val joules = result.displayJoules
        val usesTotal = result.displaysTotalEnergy
        binding.energyPerRoundMetric.setMetricData(
            R.drawable.ic_energy_bolt,
            context.getString(R.string.benchmark_energy_per_round_label),
            if (joules != null) context.getString(R.string.benchmark_energy_joules,
                joules / result.samples.size) else "—",
            context.getString(if (usesTotal) R.string.benchmark_energy_whole_device_energy
                else R.string.benchmark_energy_net_energy),
            R.color.benchmark_warning
        )
        binding.energyAveragePowerMetric.setMetricData(
            R.drawable.ic_gauge,
            context.getString(R.string.benchmark_energy_average_power_label),
            if (joules != null) context.getString(R.string.benchmark_energy_watts, result.displayWatts) else "—",
            context.getString(if (usesTotal) R.string.benchmark_energy_whole_device_power
                else R.string.benchmark_energy_net_power),
            R.color.benchmark_success
        )
        binding.energyStatus.isVisible = reason != null && !usesTotal
        if (binding.energyStatus.isVisible) {
            binding.energyStatus.text = reasonText(requireNotNull(reason))
        }
        binding.energyDetails.removeAllViews()
        details(test).forEach { (label, value) ->
            val row = ViewBenchmarkEnergyDetailRowBinding.inflate(LayoutInflater.from(context), binding.energyDetails, false)
            row.detailLabel.text = label
            row.detailValue.text = value
            binding.energyDetails.addView(row.root)
        }
        setDetailsExpanded(false)
    }

    private fun setDetailsExpanded(expanded: Boolean) {
        binding.energyDetails.isVisible = expanded
        binding.energyDetailsToggle.setIconResource(if (expanded) R.drawable.baseline_arrow_drop_up_24
            else R.drawable.baseline_arrow_drop_down_24)
        ViewCompat.setStateDescription(binding.energyDetailsToggle, context.getString(if (expanded)
            R.string.benchmark_energy_details_expanded else R.string.benchmark_energy_details_collapsed))
    }

    private fun reasonText(reason: EnergyUnavailableReason): String = context.getString(when (reason) {
        EnergyUnavailableReason.CHARGING -> R.string.benchmark_energy_charging
        EnergyUnavailableReason.INVALID_VOLTAGE -> R.string.benchmark_energy_invalid_voltage
        EnergyUnavailableReason.INVALID_CURRENT -> R.string.benchmark_energy_invalid_current
        EnergyUnavailableReason.NO_CURRENT -> R.string.benchmark_energy_no_current
        EnergyUnavailableReason.INTERRUPTED -> R.string.benchmark_energy_interrupted
        EnergyUnavailableReason.BELOW_BASELINE -> R.string.benchmark_energy_below_baseline
        EnergyUnavailableReason.TOO_SHORT, EnergyUnavailableReason.INCOMPLETE -> R.string.benchmark_energy_too_short
        EnergyUnavailableReason.UNSUPPORTED -> R.string.benchmark_energy_unsupported
    })

    private fun details(test: TestInstance): List<Pair<String, String>> = buildList {
        val result = requireNotNull(test.energy)
        fun row(label: Int, value: String) { add(context.getString(label) to value) }
        row(R.string.benchmark_energy_detail_rounds, result.samples.size.toString())
        row(R.string.benchmark_energy_detail_input_tokens, (test.nPrompt.toLong() * result.samples.size).toString())
        row(R.string.benchmark_energy_detail_output_tokens, (test.nGenerate.toLong() * result.samples.size).toString())
        row(R.string.benchmark_energy_detail_duration,
            context.getString(R.string.benchmark_energy_seconds, result.inference.seconds))
        result.inference.joules?.let {
            row(R.string.benchmark_energy_detail_gross, context.getString(R.string.benchmark_energy_joules, it))
        }
        result.baselineWatts?.let {
            row(R.string.benchmark_energy_detail_baseline, context.getString(R.string.benchmark_energy_watts, it))
        }
        if (result.reason == null) {
            row(R.string.benchmark_energy_detail_net,
                context.getString(R.string.benchmark_energy_joules, result.incrementalJoules))
        }
    }
}
