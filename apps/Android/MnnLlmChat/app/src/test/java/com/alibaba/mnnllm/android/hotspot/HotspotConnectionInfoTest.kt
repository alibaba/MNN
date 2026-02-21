package com.alibaba.mnnllm.android.hotspot

import org.junit.Assert.assertEquals
import org.junit.Test

class HotspotConnectionInfoTest {

    // ========== wifiQrContent Tests ==========

    @Test
    fun `wifiQrContent uses standard WIFI QR format`() {
        val info = HotspotConnectionInfo(
            ssid = "MyNetwork",
            password = "secret123",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:MyNetwork;T:WPA;P:secret123;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes backslash in ssid and password`() {
        val info = HotspotConnectionInfo(
            ssid = "Net\\work",
            password = "pass\\word",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:Net\\\\work;T:WPA;P:pass\\\\word;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes semicolon in ssid and password`() {
        val info = HotspotConnectionInfo(
            ssid = "Net;work",
            password = "pass;word",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:Net\\;work;T:WPA;P:pass\\;word;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes comma in ssid and password`() {
        val info = HotspotConnectionInfo(
            ssid = "Net,work",
            password = "pass,word",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:Net\\,work;T:WPA;P:pass\\,word;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes double-quote in ssid and password`() {
        val info = HotspotConnectionInfo(
            ssid = "Net\"work",
            password = "pass\"word",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:Net\\\"work;T:WPA;P:pass\\\"word;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes colon in ssid and password`() {
        val info = HotspotConnectionInfo(
            ssid = "Net:work",
            password = "pass:word",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:Net\\:work;T:WPA;P:pass\\:word;;", info.wifiQrContent)
    }

    @Test
    fun `wifiQrContent escapes multiple special characters`() {
        val info = HotspotConnectionInfo(
            ssid = "My;Net:work",
            password = "p\\a\"s,s",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("WIFI:S:My\\;Net\\:work;T:WPA;P:p\\\\a\\\"s\\,s;;", info.wifiQrContent)
    }

    // ========== urlQrContent Tests ==========

    @Test
    fun `urlQrContent formats http URL with gateway and port`() {
        val info = HotspotConnectionInfo(
            ssid = "MyNetwork",
            password = "secret",
            gatewayIp = "192.168.43.1",
            port = 8080,
        )
        assertEquals("http://192.168.43.1:8080", info.urlQrContent)
    }

    @Test
    fun `urlQrContent uses correct port`() {
        val info = HotspotConnectionInfo(
            ssid = "MyNetwork",
            password = "secret",
            gatewayIp = "10.0.0.1",
            port = 9090,
        )
        assertEquals("http://10.0.0.1:9090", info.urlQrContent)
    }
}
