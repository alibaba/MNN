package com.alibaba.mnnllm.android.debug

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class MarketDumperPluginTest {

    private class FakeController : MarketDebugController {
        var allowNetworkValue: Boolean = true
        var refreshCalled: Boolean = false
        var marketEnv: String = "prod"
        var marketUrl: String = "https://meta.alicdn.com/data/mnn/apis/model_market.json"

        override fun getStatus(): MarketDebugStatus {
            return MarketDebugStatus(
                allowNetwork = allowNetworkValue,
                loadingState = "IDLE",
                hasMarketData = true,
                modelCount = 3,
                version = "7",
                environment = marketEnv,
                networkUrl = marketUrl
            )
        }

        override fun setAllowNetwork(enabled: Boolean) {
            allowNetworkValue = enabled
        }

        override fun refreshFromNetwork(): MarketDebugStatus? {
            refreshCalled = true
            return getStatus()
        }

        override fun setEnvironment(environment: String) {
            marketEnv = environment
            marketUrl = if (environment == "dev") {
                "https://meta.alicdn.com/data/mnn/apis/model_market_dev.json"
            } else {
                "https://meta.alicdn.com/data/mnn/apis/model_market.json"
            }
        }
    }

    @Test
    fun `plugin name should be market`() {
        val plugin = MarketDumperPlugin(FakeController())
        assertEquals("market", plugin.name)
    }

    @Test
    fun `no args should print usage`() {
        val plugin = MarketDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(emptyList(), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Usage: dumpapp market"))
        assertTrue(output.contains("status"))
        assertTrue(output.contains("allow on|off"))
        assertTrue(output.contains("env dev|prod"))
        assertTrue(output.contains("refresh"))
    }

    @Test
    fun `allow command should update flag`() {
        val controller = FakeController()
        val plugin = MarketDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("allow", "off"), PrintStream(out))
        assertEquals(false, controller.allowNetworkValue)

        plugin.execute(listOf("allow", "on"), PrintStream(out))
        assertEquals(true, controller.allowNetworkValue)
    }

    @Test
    fun `refresh command should invoke controller`() {
        val controller = FakeController()
        val plugin = MarketDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("refresh"), PrintStream(out))

        assertTrue(controller.refreshCalled)
        assertTrue(out.toString().contains("Refresh success"))
    }

    @Test
    fun `env command should switch to dev and prod`() {
        val controller = FakeController()
        val plugin = MarketDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("env", "dev"), PrintStream(out))
        assertEquals("dev", controller.marketEnv)

        plugin.execute(listOf("env", "prod"), PrintStream(out))
        assertEquals("prod", controller.marketEnv)
    }

    @Test
    fun `status should include env and url`() {
        val controller = FakeController()
        val plugin = MarketDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("status"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("environment: prod"))
        assertTrue(output.contains("network_url: https://meta.alicdn.com/data/mnn/apis/model_market.json"))
    }
}
