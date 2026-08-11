// Created for issue #4259: guard against config path / custom_config merge regressions.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.model.ModelUtils
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.File
import java.io.PrintStream

/**
 * Stetho DumperPlugin for validating model config loading.
 *
 * Guards against issue #4259: when opening settings from home, config path must be
 * path to config.json (not model directory). Otherwise loadMergedConfig fails, fallback
 * to defaultConfig, and saving corrupts custom_config.json causing chat crash.
 *
 * Usage:
 *   dumpapp config validate <modelId>  - validate config path and loadMergedConfig
 */
class ConfigDumperPlugin : DumperPlugin {

    override fun getName(): String = "config"

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "validate" -> handleValidate(writer, args.getOrNull(1))
            "dump" -> handleDump(writer, args.getOrNull(1))
            else -> doUsage(writer)
        }
    }

    private fun handleDump(writer: PrintStream, modelId: String?) {
        if (modelId.isNullOrBlank()) {
            writer.println("ERROR: modelId required")
            writer.println("USAGE: dumpapp config dump <modelId>")
            writer.println("RESULT=FAIL")
            return
        }

        val configPath = ModelUtils.getConfigPathForModel(modelId)
        if (configPath.isNullOrBlank()) {
            writer.println("ERROR: getConfigPathForModel returned null/empty for modelId=$modelId")
            writer.println("RESULT=FAIL")
            return
        }

        val extraConfigPath = ModelConfig.getExtraConfigFile(modelId)
        val merged = ModelConfig.loadMergedConfig(configPath, extraConfigPath)
        if (merged == null) {
            writer.println("ERROR: loadMergedConfig returned null")
            writer.println("RESULT=FAIL")
            return
        }

        writer.println("config_path=$configPath")
        writer.println("extra_config=$extraConfigPath")
        writer.println("llm_model=${merged.llmModel ?: "(null)"}")
        writer.println("llm_weight=${merged.llmWeight ?: "(null)"}")
        writer.println("system_prompt=${merged.systemPrompt?.take(80)?.replace("\n", " ") ?: "(null)"}")
        writer.println("backend_type=${merged.backendType ?: "(null)"}")

        if (merged.llmModel.isNullOrBlank() && merged.llmWeight.isNullOrBlank()) {
            writer.println("ERROR: merged config has empty llm_model and llm_weight (corrupted)")
            writer.println("RESULT=FAIL")
            return
        }

        writer.println("RESULT=OK")
    }

    private fun handleValidate(writer: PrintStream, modelId: String?) {
        if (modelId.isNullOrBlank()) {
            writer.println("ERROR: modelId required")
            writer.println("USAGE: dumpapp config validate <modelId>")
            writer.println("RESULT=FAIL")
            return
        }

        val configPath = ModelUtils.getConfigPathForModel(modelId)
        if (configPath.isNullOrBlank()) {
            writer.println("ERROR: getConfigPathForModel returned null/empty for modelId=$modelId")
            writer.println("RESULT=FAIL")
            return
        }

        val configFile = File(configPath)
        if (configFile.isDirectory) {
            writer.println("ERROR: config path is a directory (expected config.json file): $configPath")
            writer.println("This guards against issue #4259: settings must use getConfigPathForModel, not localPath")
            writer.println("RESULT=FAIL")
            return
        }

        if (!configFile.exists()) {
            writer.println("ERROR: config file does not exist: $configPath")
            writer.println("RESULT=FAIL")
            return
        }

        val extraConfigPath = ModelConfig.getExtraConfigFile(modelId)
        val merged = ModelConfig.loadMergedConfig(configPath, extraConfigPath)
        if (merged == null) {
            writer.println("ERROR: loadMergedConfig returned null for $configPath + $extraConfigPath")
            writer.println("RESULT=FAIL")
            return
        }

        if (merged.llmModel.isNullOrBlank() && merged.llmWeight.isNullOrBlank()) {
            writer.println("WARN: merged config has empty llm_model and llm_weight (may cause native crash)")
        }

        writer.println("config_path=$configPath")
        writer.println("extra_config=$extraConfigPath")
        writer.println("RESULT=OK")
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp config <command>")
        writer.println("  validate <modelId>  - validate config path is file (not directory) and loadMergedConfig succeeds")
        writer.println("  dump <modelId>      - dump merged config (config_path, llm_model, llm_weight, system_prompt); RESULT=FAIL if corrupted")
    }
}
