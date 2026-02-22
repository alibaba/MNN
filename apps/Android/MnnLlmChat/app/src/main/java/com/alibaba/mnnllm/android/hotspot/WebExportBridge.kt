package com.alibaba.mnnllm.android.hotspot

import android.app.Activity
import android.content.Intent
import android.webkit.JavascriptInterface
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment

class WebExportBridge(private val fragment: Fragment) {

    private var pendingContent: String? = null

    val createDocumentLauncher: ActivityResultLauncher<Intent> =
        fragment.registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data ?: return@registerForActivityResult
                val content = pendingContent ?: return@registerForActivityResult
                fragment.requireContext().contentResolver.openOutputStream(uri)?.use { stream ->
                    stream.write(content.toByteArray(Charsets.UTF_8))
                }
                pendingContent = null
            }
        }

    @JavascriptInterface
    fun exportText(text: String) {
        fragment.requireActivity().runOnUiThread {
            pendingContent = text
            val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "text/plain"
                putExtra(Intent.EXTRA_TITLE, "chat-export-${java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date())}.txt")
            }
            createDocumentLauncher.launch(intent)
        }
    }
}