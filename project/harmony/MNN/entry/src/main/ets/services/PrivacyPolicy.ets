import { I18n } from './I18n';

export const PRIVACY_POLICY_KEY: string = 'privacy_policy_version';
// Bump this whenever the policy text changes, so existing users must agree again.
export const PRIVACY_POLICY_VERSION: number = 2;
// Replace with the real support mailbox before publishing to AppGallery.
export const PRIVACY_CONTACT_EMAIL: string = '待填写：联系邮箱';

const PRIVACY_POLICY_ZH: string =
  'MNN Chat 是一款在您的设备本地运行大语言模型的应用。我们重视您的个人信息和隐私保护，' +
  '并按照法律法规要求和业界安全标准处理您的信息。\n\n' +
  '一、本地数据\n' +
  '聊天记录、会话附件、模型文件和运行配置仅保存在本机应用沙箱内，不会上传到我们的服务器；' +
  '卸载应用会删除全部本地数据。\n\n' +
  '二、模型下载\n' +
  '当您从模型市场下载模型时，应用会直接访问 ModelScope、HuggingFace 或 Modelers 的公开接口。' +
  '如果配置了私有仓库令牌，令牌只保存在本机，仅用于向对应平台发起请求。' +
  'HuggingFace 服务器位于境外，使用该来源时相关网络请求信息由该平台按其隐私政策处理。\n\n' +
  '三、排行榜（仅在您主动提交成绩时）\n' +
  '应用会上传本机随机生成的匿名用户 ID（可在应用内修改）、设备型号、芯片、内存、' +
  'Prefill/Decode 速度与峰值内存到排行榜服务 mnn-mnnchatleaderboard.ms.show（服务器位于中国境内），' +
  '用于生成和展示性能排名。不会上传聊天内容或附件。您可以通过下方邮箱申请删除您的排行榜数据。\n\n' +
  '四、语音输入\n' +
  '当您主动点击「语音」或「录音」时，应用会使用设备内置的离线语音识别能力在本地处理麦克风音频，' +
  '音频不会上传；仅识别后的文字会进入对话。\n\n' +
  '五、本地 API 服务\n' +
  '本地 API 默认只监听 127.0.0.1，由您手动启动；您可以手动开启局域网访问并设置 Bearer 鉴权。' +
  '请求内容仅在本机处理，不经过我们的服务器。\n\n' +
  '六、设备权限\n' +
  '网络权限用于模型市场、模型下载与排行榜；麦克风权限仅在您主动语音输入或录音时使用；' +
  '后台运行权限仅在您主动开始下载或启动本地 API 服务后，用于后台继续传输/响应，' +
  '系统会显示常驻通知，您可随时停止。相册、文件与拍照通过系统选择器完成，应用不申请存储或相机权限。' +
  '您可以在系统设置中随时关闭上述权限。\n\n' +
  '七、第三方 SDK 与共享\n' +
  '应用使用华为 HMS Core 提供的系统能力（离线语音识别、系统分享、保存图片），' +
  '相关数据在设备内或由系统处理。应用不含广告、统计或崩溃上报 SDK，不出售您的个人信息。' +
  '除模型平台与排行榜服务外，我们不会向第三方提供您的个人信息。\n\n' +
  '八、您的权利\n' +
  '您可以查看和复制聊天记录与配置，重命名会话、修改排行榜用户 ID，删除会话、模型或清理缓存；' +
  '卸载应用会删除全部本地数据。您可以关闭麦克风权限或停止使用排行榜来撤回同意。' +
  '本应用没有账号体系，不涉及账户注销。\n\n' +
  '九、存储地点与期限\n' +
  '本地数据保存在您的设备上，直至您删除或卸载；排行榜数据保存在中国境内服务器，' +
  '保留至您申请删除或服务停止（最长不超过 24 个月）。除您主动使用 HuggingFace 来源外，' +
  '本应用不向境外传输您的个人信息。\n\n' +
  '十、联系我们\n' +
  '如您对本声明有任何疑问或希望行使相关权利，请联系：' + PRIVACY_CONTACT_EMAIL +
  '。我们将在 15 个工作日内回复。您也可以通过 https://github.com/alibaba/MNN/issues 反馈。\n\n' +
  '生效日期：2026-11-11。继续使用即表示您已阅读并同意本隐私声明。';

const PRIVACY_POLICY_EN: string =
  'MNN Chat runs large language models locally on your device. We respect your privacy and handle ' +
  'your information in line with applicable laws and industry security practices.\n\n' +
  '1. Local data\n' +
  'Chats, attachments, model files and runtime settings stay in the app sandbox on this device and ' +
  'are never uploaded to our servers. Uninstalling the app removes all local data.\n\n' +
  '2. Model downloads\n' +
  'When you download a model, the app calls the public APIs of ModelScope, HuggingFace or Modelers ' +
  'directly. Private repository tokens, if configured, are stored only on this device and used ' +
  'solely for requests to that platform. HuggingFace servers are located outside China; when you ' +
  'use that source, request metadata is handled by that platform under its own privacy policy.\n\n' +
  '3. Leaderboard (only when you submit a score)\n' +
  'The app uploads a locally generated anonymous user ID (editable in the app), device model, ' +
  'chipset, memory size, prefill/decode speed and peak memory to the leaderboard service ' +
  'mnn-mnnchatleaderboard.ms.show (servers in China) to produce the ranking. Chat content and ' +
  'attachments are never uploaded. You may email us to delete your leaderboard data.\n\n' +
  '4. Voice input\n' +
  'When you tap Voice or Record, on-device offline speech recognition processes microphone audio ' +
  'locally; the audio is not uploaded. Only the recognized text is inserted into the chat.\n\n' +
  '5. Local API server\n' +
  'The local API listens on 127.0.0.1 by default and is started by you. You may enable LAN access ' +
  'and Bearer authentication manually. Requests are processed on this device and never reach our ' +
  'servers.\n\n' +
  '6. Device permissions\n' +
  'Network is used for the model market, downloads and the leaderboard. Microphone is used only ' +
  'when you actively use voice input or recording. Background running is requested only after you ' +
  'start a download or the local API server, so it can continue in the background; the system shows ' +
  'a persistent notification and you can stop it at any time. Gallery, files and photos are picked ' +
  'through system pickers; the app does not request storage or camera permissions. You can revoke ' +
  'these permissions in system settings.\n\n' +
  '7. Third-party SDKs and sharing\n' +
  'The app uses Huawei HMS Core system capabilities (offline speech recognition, system share, ' +
  'saving images), which process data on the device or within the system. The app contains no ' +
  'advertising, analytics or crash-reporting SDK and never sells your personal information. Apart ' +
  'from model platforms and the leaderboard service, we do not provide your information to third ' +
  'parties.\n\n' +
  '8. Your rights\n' +
  'You can view and copy chats and configuration, rename sessions, edit the leaderboard user ID, ' +
  'delete sessions or models, and clear caches. Uninstalling the app removes all local data. You ' +
  'can withdraw consent by revoking microphone permission or stopping use of the leaderboard. The ' +
  'app has no account system, so account deletion does not apply.\n\n' +
  '9. Storage and retention\n' +
  'Local data stays on your device until you delete it or uninstall the app. Leaderboard data is ' +
  'stored on servers in China until you request deletion or the service ends (at most 24 months). ' +
  'Except when you actively use the HuggingFace source, the app does not transfer personal ' +
  'information outside China.\n\n' +
  '10. Contact us\n' +
  'For questions or to exercise your rights, contact: ' + PRIVACY_CONTACT_EMAIL +
  '. We reply within 15 business days. You can also file feedback at ' +
  'https://github.com/alibaba/MNN/issues.\n\n' +
  'Effective date: 2026-11-11. By continuing to use the app you confirm that you have read and ' +
  'agree to this privacy statement.';

export function privacyPolicyText(): string {
  return I18n.isEnglish() ? PRIVACY_POLICY_EN : PRIVACY_POLICY_ZH;
}