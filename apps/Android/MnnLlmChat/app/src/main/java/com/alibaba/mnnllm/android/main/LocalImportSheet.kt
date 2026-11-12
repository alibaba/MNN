package com.alibaba.mnnllm.android.main

import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.alibaba.mnnllm.android.databinding.SheetLocalImportBinding
import com.alibaba.mnnllm.android.modelist.LocalModelsProvider
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.io.File

class LocalImportSheet : BottomSheetDialogFragment() {

    private var _binding: SheetLocalImportBinding? = null
    private val binding get() = _binding!!

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = super.onCreateDialog(savedInstanceState) as BottomSheetDialog
        dialog.setOnShowListener {
            val sheet = dialog.findViewById<View>(com.google.android.material.R.id.design_bottom_sheet)
            sheet?.let { BottomSheetBehavior.from(it).state = BottomSheetBehavior.STATE_EXPANDED }
        }
        return dialog
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = SheetLocalImportBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        val activity = requireActivity() as MainActivity
        val dropInDir = LocalModelsProvider.dropInModelsDir(requireContext())?.absolutePath
            ?: File(requireContext().filesDir, LocalModelsProvider.DROP_IN_DIR_NAME).absolutePath
        val adbCommand = "adb shell mkdir -p /data/local/tmp/mnn_models && adb push \${model_path} /data/local/tmp/mnn_models/"

        binding.tvDropPath.text = dropInDir
        binding.tvAdbCommand.text = adbCommand

        val pick = View.OnClickListener {
            dismiss()
            activity.launchImportFolderPicker()
        }
        binding.cardPickFolder.setOnClickListener(pick)
        binding.btnPickFolder.setOnClickListener(pick)
        binding.btnCopyPath.setOnClickListener { activity.copyImportText("Model directory", dropInDir) }
        binding.btnCopyAdb.setOnClickListener { activity.copyImportText("ADB Command", adbCommand) }
        binding.btnAdvanced.setOnClickListener {
            val open = binding.groupAdb.visibility != View.VISIBLE
            binding.groupAdb.visibility = if (open) View.VISIBLE else View.GONE
            binding.btnAdvanced.setIconResource(
                if (open) com.alibaba.mnnllm.android.R.drawable.ic_expand_less else com.alibaba.mnnllm.android.R.drawable.ic_expand_more
            )
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
