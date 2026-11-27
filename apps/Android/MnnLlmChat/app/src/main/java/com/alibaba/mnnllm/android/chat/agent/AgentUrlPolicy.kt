package com.alibaba.mnnllm.android.chat.agent

import okhttp3.Dns
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import java.net.Inet4Address
import java.net.Inet6Address
import java.net.InetAddress
import java.net.UnknownHostException
import java.util.Locale

class AgentUrlPolicy {
    fun validate(raw: String): HttpUrl {
        require(raw.length in 1..2048 && raw.none { it.isWhitespace() || it.code < 32 || it == '\\' }) { "Invalid URL" }
        require(raw.startsWith("https://", true) || raw.startsWith("http://", true)) { "Only HTTP(S) URLs are allowed" }
        val url = requireNotNull(raw.toHttpUrlOrNull()) { "Invalid URL" }
        val authority = raw.substringAfter("://").substringBefore('/').substringBefore('?').substringBefore('#')
        require(url.username.isEmpty() && url.password.isEmpty() && '@' !in authority) { "URL credentials are forbidden" }
        require(url.port == if (url.isHttps) 443 else 80) { "Only default HTTP(S) ports are allowed" }
        validateHost(url.host)
        return url.newBuilder().fragment(null).build()
    }

    fun validateHost(rawHost: String) {
        val host = rawHost.trimEnd('.').lowercase(Locale.ROOT).removePrefix("[").removeSuffix("]")
        require(host.isNotEmpty() && !host.contains('%')) { "Invalid host" }
        require(host != "localhost" && listOf(".localhost", ".local", ".lan", ".internal", ".home", ".onion")
            .none { host.endsWith(it) }) { "Local hosts are forbidden" }
        if (host.contains(':') || host.all { it in '0'..'9' || it == '.' }) {
            val address = try { InetAddress.getByName(host) } catch (_: Exception) { null }
            require(address != null && isPublicAddress(address)) { "Non-public address is forbidden" }
        } else {
            require(host.contains('.') && host.split('.').all { label ->
                label.isNotEmpty() && label.length <= 63 && label.first() != '-' && label.last() != '-' &&
                    label.all { it in 'a'..'z' || it in '0'..'9' || it == '-' }
            }) { "Invalid public host" }
            require(!Regex("^(?:0x[0-9a-f]+|[0-9]+)(?:\\.(?:0x[0-9a-f]+|[0-9]+))*$").matches(host)) { "Numeric host is forbidden" }
        }
    }

    fun isPublicAddress(address: InetAddress): Boolean {
        if (address.isAnyLocalAddress || address.isLoopbackAddress || address.isLinkLocalAddress ||
            address.isSiteLocalAddress || address.isMulticastAddress) return false
        val bytes = address.address.map { it.toInt() and 255 }
        if (address is Inet4Address) {
            val a = bytes[0]; val b = bytes[1]; val c = bytes[2]
            return !(a == 0 || a == 10 || a == 127 || a >= 224 ||
                (a == 100 && b in 64..127) || (a == 169 && b == 254) || (a == 172 && b in 16..31) ||
                (a == 192 && (b == 168 || (b == 0 && c in 0..2) || (b == 88 && c == 99))) ||
                (a == 198 && (b in 18..19 || (b == 51 && c == 100))) || (a == 203 && b == 0 && c == 113))
        }
        if (address is Inet6Address) {
            if (bytes[0] !in 0x20..0x3f) return false
            if (bytes[0] == 0x20 && bytes[1] == 0x02) return false
            if (bytes[0] == 0x20 && bytes[1] == 0x01 &&
                ((bytes[2] == 0x0d && bytes[3] == 0xb8) || bytes[2] < 2)) return false
            return true
        }
        return false
    }

    fun urlsInUserText(text: String): Set<String> = Regex("https?://[^\\s<>\"'\\[\\]]+", RegexOption.IGNORE_CASE)
        .findAll(text).take(32).mapNotNull { match ->
            try { validate(match.value.trimEnd('.', ',', '。', '，', ')', '）', ';', '；')).toString() }
            catch (_: IllegalArgumentException) { null }
        }.toSet()
}

class AgentPublicDns(
    private val policy: AgentUrlPolicy = AgentUrlPolicy(),
    private val resolver: Dns = Dns.SYSTEM,
) : Dns {
    override fun lookup(hostname: String): List<InetAddress> {
        try { policy.validateHost(hostname) } catch (_: IllegalArgumentException) {
            throw UnknownHostException("Non-public host is forbidden")
        }
        val addresses = resolver.lookup(hostname)
        if (addresses.isEmpty() || addresses.any { !policy.isPublicAddress(it) }) {
            throw UnknownHostException("DNS returned a non-public address")
        }
        return addresses.toList()
    }
}
