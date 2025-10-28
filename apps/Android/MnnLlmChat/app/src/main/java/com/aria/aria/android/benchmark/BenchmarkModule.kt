// Created by ruoyi.sjd on 2025/5/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.benchmark

import android.widget.Toast
import androidx.fragment.app.FragmentActivity
import com.alibaba.mnnllm.android.R
import com.google.zxing.integration.android.IntentIntegrator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class BenchmarkModule(private val activity: FragmentActivity) {

    private val testUpload = false
    private val benchmarkDatasetRoot = "/data/local/tmp/mnn_bench/datasets"
    private val maxDataItemCount = 3
    private var currentDataItemCount = 0

    fun start(waitForLastCompleted: suspend () -> Unit, handleSendMessage: suspend (String) -> HashMap<String, Any>) {
        val selectDataSetFragment = SelectDataSetFragment()
        selectDataSetFragment.setOnItemClickListener { optionItem ->
            handleDatasetOptionClick(optionItem, waitForLastCompleted, handleSendMessage)
            selectDataSetFragment.dismiss()
        }
        selectDataSetFragment.show(activity.supportFragmentManager, SelectDataSetFragment.TAG)
    }

    val enabled
        get() = (File(benchmarkDatasetRoot).listFiles()?.size ?: 0) > 0

    private fun startQRScanner() {
        IntentIntegrator(activity).apply {
            setOrientationLocked(false)
            setPrompt("Scan compare‑URL QR code")
        }.initiateScan()
    }

    private fun handleDatasetOptionClick(optionItem: DatasetOptionItem,
                                         waitForLastCompleted: suspend () -> Unit,
                                         handleSendMessage: suspend (String) -> HashMap<String, Any>) {
        if (testUpload) {
            startQRScanner()
            return
        }
        currentDataItemCount = 0
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val lines = withContext(Dispatchers.IO) {
                    val file = File("${benchmarkDatasetRoot}/${optionItem.id}.jsonl")
                    if (!file.exists()) {
                        throw Exception("File not exits: ${file.path}")
                    }
                    val fileLines = file.readLines()
                    if (fileLines.isEmpty()) {
                        throw Exception("file empty")
                    }
                    fileLines
                }
                for (line in lines) {
                    val jsonObject = try {
                        JSONObject(line)
                    } catch (e: Exception) {
                        throw Exception("Invalid json: $line")
                    }
                    waitForLastCompleted()
                    if (currentDataItemCount >= maxDataItemCount) {
                        break
                    }
                    val result = handleSendMessage(jsonObject.getString("prompt"))
                    currentDataItemCount++
                }
                Toast.makeText(activity, activity.getString(R.string.dataset_process_complete, optionItem.title), Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(activity, activity.getString(R.string.error_on_process_dataset, e.message), Toast.LENGTH_SHORT).show()
            }
        }
    }

}