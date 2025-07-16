# 网络调试指南 / Network Debug Guide

## 排行榜上传调试 / Leaderboard Upload Debugging

### 1. 功能开关 / Feature Toggle

在 `BenchmarkPresenter.kt` 中可以切换功能：

```kotlin
// 设置为 true 使用排行榜上传，false 使用分享功能
private val useLeaderboardUpload = true
```

### 2. 日志标签 / Log Tags

使用以下标签过滤日志：

```bash
# 主要的网络请求日志
adb logcat -s LeaderboardService

# HTTP 详细日志 (包含请求头、响应体等)
adb logcat | grep "HTTP:"

# Benchmark相关日志
adb logcat -s BenchmarkPresenter BenchmarkFragment
```

### 3. 网络调试步骤 / Network Debug Steps

#### 步骤1: 检查网络连接
```bash
# 检查设备网络连接
adb shell ping -c 3 mnn-mnnchatleaderboard.ms.show
```

#### 步骤2: 查看详细日志
运行应用并执行上传，查看以下关键日志（现在使用emoji图标更易识别）：

1. **设备信息获取**
   ```
   LeaderboardService: 📱 ===== COLLECTING DEVICE INFO =====
   LeaderboardService: 📱 Device Details:
   LeaderboardService:    🏷️  Model: [设备型号]
   LeaderboardService:    🔧 Chipset: [芯片组]
   LeaderboardService:    💿 RAM: [内存] MB
   LeaderboardService: ✅ Device info collected successfully
   ```

2. **提交分数请求**
   ```
   LeaderboardService: 🚀 ===== SUBMITTING SCORE TO LEADERBOARD =====
   LeaderboardService: 📡 URL: https://mnn-mnnchatleaderboard.ms.show/gradio_api/call/submit_score
   LeaderboardService: 👤 User ID: [用户ID]
   LeaderboardService: 🤖 Model: [模型名称]
   LeaderboardService: 📊 Prefill Speed: [速度] tokens/s
   LeaderboardService: ⚡ Decode Speed: [速度] tokens/s
   LeaderboardService: 📤 Request JSON: [格式化的JSON]
   LeaderboardService: 🏷️  Status: [HTTP状态码]
   LeaderboardService: 📄 Response Body: [格式化的响应]
   ```

3. **成功提交**
   ```
   LeaderboardService: 🎉 ===== SCORE SUBMITTED SUCCESSFULLY =====
   LeaderboardService: ✅ Status: Success
   LeaderboardService: 🏆 Your benchmark score has been uploaded to the leaderboard!
   ```

4. **获取排名请求**
   ```
   LeaderboardService: 🏅 ===== GETTING USER RANKING =====
   LeaderboardService: ✅ Event ID extracted: [事件ID]
   LeaderboardService: 🔄 Starting result polling...
   LeaderboardService: 📊 ===== POLLING RANKING RESULT =====
   LeaderboardService: 🎯 ===== RANKING RESULTS =====
   LeaderboardService: 🏆 Your Rank: [排名]
   LeaderboardService: 👥 Total Users: [总用户数]
   ```

#### 步骤3: 常见问题诊断

**问题1: 网络连接失败**
- 日志特征: `ConnectException`, `UnknownHostException`
- 解决方案: 检查网络连接，确认防火墙设置

**问题2: HTTP 错误状态码**
- 日志特征: `Response code: 4xx/5xx`
- 解决方案: 检查API端点是否正确，服务器是否正常

**问题3: 请求格式错误**
- 日志特征: `400 Bad Request`, `422 Unprocessable Entity`
- 解决方案: 检查请求JSON格式是否正确
- 常见原因: 数组格式错误（已修复：使用JSONArray而不是arrayOf）

**问题4: 响应解析失败**
- 日志特征: `Error parsing rank data`
- 解决方案: 检查API响应格式是否符合预期

### 4. 手动测试API / Manual API Testing

可以使用curl命令手动测试API：

#### 测试提交分数
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"data": ["{\"user_id\":\"test_user\",\"llm_model\":\"Test-Model\",\"device_model\":\"Test Device\",\"device_chipset\":\"Test Chip\",\"device_memory\":8192,\"prefill_speed\":100.0,\"decode_speed\":50.0,\"memory_usage\":1024.0}"]}' \
  https://mnn-mnnchatleaderboard.ms.show/gradio_api/call/submit_score
```

#### 测试获取排名
```bash
# 步骤1: 发起排名查询
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"data": ["test_user", "Test-Model"]}' \
  https://mnn-mnnchatleaderboard.ms.show/gradio_api/call/get_my_rank

# 步骤2: 使用返回的event_id轮询结果
curl -X GET \
  https://mnn-mnnchatleaderboard.ms.show/gradio_api/call/get_my_rank/[EVENT_ID]
```

### 5. 抓包分析 / Packet Capture

如果需要更深入的网络分析，可以使用以下工具：

1. **Charles Proxy** - 在电脑上设置代理
2. **Wireshark** - 分析网络数据包
3. **Android Studio Network Inspector** - 查看应用网络请求

### 6. 常用调试命令 / Common Debug Commands

```bash
# 清除应用数据重新测试
adb shell pm clear com.alibaba.mnnllm.android

# 查看应用日志
adb logcat -s MNN

# 查看网络相关日志
adb logcat | grep -E "(HTTP|LeaderboardService|okhttp)"

# 导出日志到文件
adb logcat -s LeaderboardService > leaderboard_debug.log
```

### 7. 故障排除检查清单 / Troubleshooting Checklist

- [ ] 网络连接正常
- [ ] API URL正确
- [ ] 请求格式正确 
- [ ] 响应状态码为200
- [ ] 设备信息获取成功
- [ ] JSON解析无错误
- [ ] 日志中无异常堆栈

### 8. 联系支持 / Contact Support

如果问题仍然存在，请提供以下信息：

1. 完整的日志输出 (使用 `adb logcat -s LeaderboardService`)
2. 设备信息 (型号、Android版本)
3. 网络环境 (WiFi/4G/5G)
4. 复现步骤
5. 错误截图

---

## 已修复的问题 / Fixed Issues

### ✅ HTTP 422 错误修复
**问题**: 服务器返回 `422 Unprocessable Entity` 错误，消息为 "Input should be a valid list"
```
{"detail":[{"type":"list_type","loc":["body","data"],"msg":"Input should be a valid list","input":"[Ljava.lang.String;@24d5518"}]}
```

**原因**: 使用 `arrayOf()` 创建的Kotlin数组在序列化为JSON时变成了Java对象的字符串表示

**解决方案**: 
- ❌ 错误做法: `put("data", arrayOf(submissionData.toString()))`
- ✅ 正确做法: `put("data", JSONArray().apply { put(submissionData.toString()) })`

### 📊 日志格式改进
- 添加emoji图标，便于快速识别日志类型
- JSON格式化输出，提高可读性
- 分阶段显示，清晰展示请求/响应流程
- 详细的错误诊断和解决建议

## 开发提示 / Development Tips

- 在开发阶段，建议先设置 `useLeaderboardUpload = false` 测试分享功能
- 使用模拟器测试时，确保模拟器有网络连接
- 生产环境部署前，务必测试网络功能的稳定性
- 使用新的emoji日志可以快速定位问题所在阶段 