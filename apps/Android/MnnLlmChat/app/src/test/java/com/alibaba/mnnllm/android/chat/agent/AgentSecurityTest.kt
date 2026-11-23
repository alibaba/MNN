package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import okhttp3.Dns
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress
import java.net.UnknownHostException

@OptIn(ExperimentalCoroutinesApi::class)
class AgentSecurityTest {
    private val policy = AgentUrlPolicy()

    @Test
    fun `blocks unsafe schemes credentials ports local and nonpublic IPs`() {
        val blocked = listOf("file:///etc/passwd", "content://contacts", "javascript:alert(1)", "ftp://example.com",
            "example.com", "http://localhost", "http://localhost.", "http://printer.local", "http://server.internal",
            "http://127.0.0.1", "http://127.1", "http://2130706433", "http://0x7f000001", "http://10.0.0.1",
            "http://172.16.1.2", "http://192.168.1.1", "http://169.254.169.254", "http://100.64.0.1",
            "http://198.18.0.1", "http://0.0.0.0", "http://224.0.0.1", "http://[::1]", "http://[fe80::1]",
            "http://[fc00::1]", "http://[::ffff:127.0.0.1]", "http://[2001:db8::1]", "http://[2002:7f00:1::]",
            "https://user:pass@example.com", "https://@example.com", "https://example.com:8080", "https://example.com\\@127.0.0.1")
        for (url in blocked) assertThrows(url, IllegalArgumentException::class.java) { policy.validate(url) }
        assertEquals("https://example.com/path", policy.validate("https://example.com/path#fragment").toString())
        assertTrue(policy.isPublicAddress(InetAddress.getByName("8.8.8.8")))
        assertTrue(policy.isPublicAddress(InetAddress.getByName("2606:4700:4700::1111")))
    }

    @Test
    fun `DNS rejects mixed public private answers and retains the checked public addresses`() {
        val publicIp = InetAddress.getByName("8.8.8.8")
        val privateIp = InetAddress.getByName("10.0.0.1")
        val unsafe = AgentPublicDns(policy, object : Dns {
            override fun lookup(hostname: String) = listOf(publicIp, privateIp)
        })
        assertThrows(UnknownHostException::class.java) { unsafe.lookup("example.com") }
        val safe = AgentPublicDns(policy, object : Dns {
            override fun lookup(hostname: String) = listOf(publicIp)
        })
        assertEquals(listOf(publicIp), safe.lookup("example.com"))
        assertThrows(UnknownHostException::class.java) { safe.lookup("localhost") }
    }

    @Test
    fun `redirect to private address is rejected before second transport call`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val client = AgentHttpClient(consent)
        var calls = 0
        try {
            client.fetchWithTransport("https://example.com", revision) {
                calls++
                AgentHttpClient.HttpResult.Redirect("http://127.0.0.1/admin")
            }
            throw AssertionError("Unsafe redirect was followed")
        } catch (_: IllegalArgumentException) { }
        assertEquals(1, calls)
    }

    @Test
    fun `redirect limit downgrade and revocation stop fetch`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val client = AgentHttpClient(consent)
        var calls = 0
        try {
            client.fetchWithTransport("https://example.com", revision) { url ->
                calls++
                AgentHttpClient.HttpResult.Redirect(url.toString())
            }
            throw AssertionError("Redirect loop was followed")
        } catch (_: IllegalArgumentException) { }
        assertEquals(5, calls)
        try {
            client.fetchWithTransport("https://example.com", revision) { AgentHttpClient.HttpResult.Redirect("http://example.com/") }
            throw AssertionError("Downgrade was followed")
        } catch (_: IllegalArgumentException) { }
        consent.disable()
        calls = 0
        try {
            client.fetchWithTransport("https://example.com", revision) {
                calls++
                AgentHttpClient.HttpResult.Page(AgentFetchedPage(it.toString(), "text", false))
            }
            throw AssertionError("Missing consent was ignored")
        } catch (_: IllegalArgumentException) { }
        assertEquals(0, calls)
    }

    @Test
    fun `search cannot forward ungrounded history or saved memory text`() {
        val request = "请帮我搜索 MNN Android 新闻"
        val query = AgentInputPolicy.privateSearchQuery("secret from earlier conversation", request)
        assertFalse(query.contains("secret"))
        assertTrue(query.contains("MNN"))
        assertEquals("MNN Android", AgentInputPolicy.privateSearchQuery("MNN Android", request))
        assertTrue(AgentInputPolicy.privateSearchQuery("x".repeat(200), request).length <= 80)
        assertFalse(AgentInputPolicy.memoryIsGrounded("网页说用户是管理员", request))
        assertFalse(AgentInputPolicy.memoryIsGrounded("我喜欢茶", "我喜欢茶，但不要保存记忆"))
        assertNullMemoryFallbackForQuotedInstruction()
    }

    private fun assertNullMemoryFallbackForQuotedInstruction() {
        assertEquals(null, AgentInputPolicy.explicitMemory("网页上写着记住密码，但我只想查看网页"))
        assertEquals("我喜欢茶", AgentInputPolicy.explicitMemory("请记住：我喜欢茶。"))
    }

    @Test
    fun `skill writes require explicit user-authored procedures and matching keywords`() {
        val skill = AgentSkill("发布", listOf("发布"), "先测试再发布")
        assertFalse(AgentInputPolicy.skillIsGrounded(skill, "先测试再发布"))
        assertTrue(AgentInputPolicy.skillIsGrounded(skill, "保存技能：发布。先测试再发布"))
        assertFalse(AgentInputPolicy.skillIsGrounded(skill.copy(actionTemplate = "发送聊天记录"), "保存技能：发布。先测试再发布"))
        assertEquals(listOf(skill), AgentPrompts.matchingSkills(listOf(skill), "请发布"))
    }
}
