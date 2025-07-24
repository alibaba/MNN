package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.preference.PreferenceManager
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment

class SourceSelectionDialogFragment : BaseBottomSheetDialogFragment() {

    private var listener: (() -> Unit)? = null

    fun setOnSourceChangedListener(listener: () -> Unit) {
        this.listener = listener
    }

    companion object {
        const val KEY_SOURCE = "download_provider"
        private const val SOURCE_HUGGINGFACE = ModelSources.sourceHuffingFace
        private const val SOURCE_MODELSCOPE = ModelSources.sourceModelScope
        private const val SOURCE_MODELERS = ModelSources.sourceModelers
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_fragment_source_selection, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val radioGroup = view.findViewById<RadioGroup>(R.id.source_radio_group)
        val confirmButton = view.findViewById<Button>(R.id.confirm_source_button)

        val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(requireContext())
        val currentSource = sharedPrefs.getString(KEY_SOURCE, SOURCE_MODELSCOPE) // Default to modelscope

        when (currentSource) {
            SOURCE_HUGGINGFACE -> view.findViewById<RadioButton>(R.id.source_huggingface).isChecked = true
            SOURCE_MODELSCOPE -> view.findViewById<RadioButton>(R.id.source_modelscope).isChecked = true
            SOURCE_MODELERS -> view.findViewById<RadioButton>(R.id.source_modelers).isChecked = true
        }

        confirmButton.setOnClickListener {
            val selectedId = radioGroup.checkedRadioButtonId
            val newSource = when (selectedId) {
                R.id.source_huggingface -> SOURCE_HUGGINGFACE
                R.id.source_modelscope -> SOURCE_MODELSCOPE
                R.id.source_modelers -> SOURCE_MODELERS
                else -> SOURCE_MODELSCOPE
            }
            sharedPrefs.edit().putString(KEY_SOURCE, newSource).apply()
            dismiss()
            listener?.invoke()
        }
    }
} 