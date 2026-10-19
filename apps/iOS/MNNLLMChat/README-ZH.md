# MNN Chat iOS 应用

[View English Documentation](./README.md)

## 介绍

MNN Chat 是基于 MNN 引擎的 iOS 本地大模型应用，支持文本、图片、音频、视频等多模态对话。

模型下载到本机后，推理全部在设备上完成，对话内容不会上传。只有开启 Agent 模式并触发联网搜索时，应用才会访问网络（发送搜索关键词、读取网页）。

## 功能

1. **首次使用与模型下载**
    - 未下载任何模型时，首页展示推荐模型卡片，一键下载并开始聊天
    - 模型市场：获取 MNN 支持的模型列表，支持关键词搜索与标签筛选
    - 支持切换 ModelScope、Hugging Face、Modelers 下载源
    - 本机适配标签：按模型大小与设备内存估算"适合本机 / 可能较慢 / 超出本机内存"
    - 下载前检查存储空间；支持暂停、断点续传、取消，显示进度与实时速度，失败时行内重试
2. **模型管理**
    - 入口位于会话抽屉底部的"模型管理"，分为"已下载模型"与"下载模型"两个页签
    - 左滑删除、置顶、重命名；删除前显示可释放空间
    - 可删除正在使用的模型：先卸载模型（聊天记录保留），再删除文件并切换到其他模型
3. **聊天工作区与会话管理**
    - 打开即进入聊天页，输入框上方切换模型
    - 左侧会话抽屉：搜索历史会话、左滑删除，底部提供模型管理、设置与 Benchmark 入口
    - 切换同一模型的历史会话无需重新加载模型
4. **对话体验**：支持完整 Markdown 输出
    - 思考过程独立面板，正文开始后自动折叠；思考模式可通过输入框灯泡按钮或模型设置开关，按模型持久化
    - 消息长按菜单：复制、编辑后重发、重新生成、选择文本、分享
    - 代码块标题栏显示语言并提供复制按钮
    - 回答后显示 Prefill / Decode 速度；支持停止生成，以及仅有思考内容时"直接回答"
5. **Agent 模式**
    - 模型选择器旁的 Agent 开关，开启后模型可按需调用工具，适用于 Qwen3.5 0.8B / 2B 等小模型
    - 内置工具：联网搜索、读取搜索结果网页、获取当前时间、计算器、保存记忆
    - 工具调用过程显示在独立的可折叠面板中，与思考过程分开展示
    - 读取网页仅限搜索结果中出现过的站点，模型通过结果编号引用，避免编造链接
    - 记忆与技能可在"设置 → Agent 模式"中查看和清空
6. **上下文管理**
    - 连续对话：后续提问带入先前问答，不带入思考内容
    - 上下文用量面板：显示本轮输入、生成与总用量，达到 80% 时图标变为橙色
    - 按本机可用内存与模型 KV 缓存占用推荐最大上下文长度，可一键采用
    - 接近上限时在本机自动压缩较早对话
7. **模型设置**
    - 专业名词旁的问号说明，点击后以气泡形式展示含义、推荐值与是否需要重新加载
    - 采样类型：Mixed（默认）与 Greedy；Mixed 下可调 Penalty、Top K、TFS-Z、Typical、Top P、Min P、Temperature
    - 支持配置 mmap、后端类型（CPU / Metal）、精度、线程数、最大输出 token 数
    - 支持配置多模态 Prompt API、视频抽帧数、Omni 模型语音输出、Diffusion 参数
8. **多模态**
    - 图片：拍照或从相册选择，自动缩放、校正方向并转换 HEIC
    - 视频：按约每秒 1 帧均匀抽帧后作为图片输入
    - 音频：语音输入，Omni 模型支持语音输出
    - Diffusion：Stable Diffusion 文生图、Sana 风格迁移（如吉卜力风格）
9. **基准测试**
    - 自动化测试 Prefill 速度、Decode 速度与内存占用，支持文本、图片和音频批量测试
10. **应用设置**
    - 中英文切换即时生效；隐私说明、开源许可、帮助与反馈

### 应用预览

| | | | |
|--|--|--|--|
| **首页推荐模型** | **模型市场下载状态** | **回答展示** | **思考过程** |
| <img src="./assets/features/01-home-recommended.jpg" width="200" /> | <img src="./assets/features/03-market-download-states.jpg" width="200" /> | <img src="./assets/features/05-chat-answer.jpg" width="200" /> | <img src="./assets/features/15-thinking-streaming.jpg" width="200" /> |
| **消息长按菜单** | **会话抽屉** | **上下文用量** | **设置问号说明** |
| <img src="./assets/features/06-message-menu.jpg" width="200" /> | <img src="./assets/features/09-history-drawer.jpg" width="200" /> | <img src="./assets/features/17-context-usage-panel.jpg" width="200" /> | <img src="./assets/features/11-help-popover.jpg" width="200" /> |
| **Agent 工具调用** | **Agent 回答** | | |
| <img src="./assets/features/21-agent-tool-call.jpg" width="200" /> | <img src="./assets/features/22-agent-answer.jpg" width="200" /> | | |
