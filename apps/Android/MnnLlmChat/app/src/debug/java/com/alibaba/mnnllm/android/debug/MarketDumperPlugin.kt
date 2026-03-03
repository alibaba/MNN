package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import kotlinx.coroutines.runBlocking
import java.io.PrintStream

internal data class MarketDebugStatus(
    val allowNetwork: Boolean,
    val loadingState: String,
    val hasMarketData: Boolean,
    val modelCount: Int,
    val version: String?,
    val environment: String,
    val networkUrl: String
)

internal interface MarketDebugController {
    fun getStatus(): MarketDebugStatus
    fun setAllowNetwork(enabled: Boolean)
    fun setEnvironment(environment: String)
    fun refreshFromNetwork(): MarketDebugStatus?
}

internal object DefaultMarketDebugController : MarketDebugController {
    private const val KEY_ALLOW_NETWORK_MARKET_DATA = "debug_allow_network_market_data"

    override fun getStatus(): MarketDebugStatus {
        val context = MnnLlmApplication.getAppContext()
        val config = ModelRepository.getModelMarketDataV2()
        val modelCount = (config?.llmModels?.size ?: 0) +
            (config?.ttsModels?.size ?: 0) +
            (config?.asrModels?.size ?: 0) +
            (config?.libs?.size ?: 0)

        return MarketDebugStatus(
            allowNetwork = PreferenceUtils.getBoolean(context, KEY_ALLOW_NETWORK_MARKET_DATA, true),
            loadingState = ModelRepository.loadingStateFlow.value.name,
            hasMarketData = config != null,
            modelCount = modelCount,
            version = config?.version,
            environment = ModelRepository.getMarketEnvironment(context),
            networkUrl = ModelRepository.getMarketNetworkUrl(context)
        )
    }

    override fun setAllowNetwork(enabled: Boolean) {
        val context = MnnLlmApplication.getAppContext()
        PreferenceUtils.setBoolean(context, KEY_ALLOW_NETWORK_MARKET_DATA, enabled)
    }

    override fun setEnvironment(environment: String) {
        val context = MnnLlmApplication.getAppContext()
        ModelRepository.setMarketEnvironment(context, environment)
    }

    override fun refreshFromNetwork(): MarketDebugStatus? {
        val config = runBlocking { ModelRepository.refreshFromNetwork() } ?: return null
        val modelCount = config.llmModels.size + config.ttsModels.size + config.asrModels.size + config.libs.size
        return MarketDebugStatus(
            allowNetwork = getStatus().allowNetwork,
            loadingState = ModelRepository.loadingStateFlow.value.name,
            hasMarketData = true,
            modelCount = modelCount,
            version = config.version,
            environment = getStatus().environment,
            networkUrl = getStatus().networkUrl
        )
    }
}

internal class MarketDumperPlugin(
    private val controller: MarketDebugController = DefaultMarketDebugController
) : DumperPlugin {

    override fun getName(): String = "market"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "status" -> handleStatus(writer)
            "allow" -> handleAllow(writer, args.drop(1))
            "env" -> handleEnv(writer, args.drop(1))
            "refresh" -> handleRefresh(writer)
            else -> doUsage(writer)
        }
    }

    private fun handleStatus(writer: PrintStream) {
        val status = controller.getStatus()
        writer.println("Model market status:")
        writer.println("  allow_network: ${status.allowNetwork}")
        writer.println("  loading_state: ${status.loadingState}")
        writer.println("  has_market_data: ${status.hasMarketData}")
        writer.println("  model_count: ${status.modelCount}")
        writer.println("  version: ${status.version ?: "unknown"}")
        writer.println("  environment: ${status.environment}")
        writer.println("  network_url: ${status.networkUrl}")
    }

    private fun handleAllow(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp market allow on|off")
            return
        }

        when (args[0].lowercase()) {
            "on", "true", "1" -> {
                controller.setAllowNetwork(true)
                writer.println("allow_network set to true")
            }
            "off", "false", "0" -> {
                controller.setAllowNetwork(false)
                writer.println("allow_network set to false")
            }
            else -> writer.println("Usage: dumpapp market allow on|off")
        }
    }

    private fun handleEnv(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp market env dev|prod")
            return
        }
        when (args[0].lowercase()) {
            "dev" -> {
                controller.setEnvironment("dev")
                writer.println("market environment set to dev")
            }
            "prod" -> {
                controller.setEnvironment("prod")
                writer.println("market environment set to prod")
            }
            else -> writer.println("Usage: dumpapp market env dev|prod")
        }
    }

    private fun handleRefresh(writer: PrintStream) {
        val result = controller.refreshFromNetwork()
        if (result == null) {
            writer.println("Refresh failed")
            return
        }
        writer.println("Refresh success")
        writer.println("  version: ${result.version ?: "unknown"}")
        writer.println("  model_count: ${result.modelCount}")
        writer.println("  environment: ${result.environment}")
        writer.println("  network_url: ${result.networkUrl}")
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp market <command>")
        writer.println("Commands:")
        writer.println("  status              - Show current market loading status")
        writer.println("  allow on|off        - Toggle network market data fetch")
        writer.println("  env dev|prod        - Switch market endpoint environment")
        writer.println("  refresh             - Force refresh market data from network")
        writer.println("Examples:")
        writer.println("  dumpapp market status")
        writer.println("  dumpapp market allow off")
        writer.println("  dumpapp market env dev")
        writer.println("  dumpapp market refresh")
    }
}
