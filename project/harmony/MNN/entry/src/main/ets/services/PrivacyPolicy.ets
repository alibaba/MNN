import { I18n } from './I18n';

export const PRIVACY_POLICY_KEY: string = 'privacy_policy_version';
export const PRIVACY_POLICY_VERSION: number = 1;

const PRIVACY_POLICY_ZH: string =
  'MNN Chat 是一款完全在本机运行的大模型客户端，我们重视你的隐私。\n\n' +
  '1. 数据存储\n' +
  '聊天记录、会话附件、模型文件和运行配置仅保存在本机应用沙箱内，不会上传到任何服务器。\n\n' +
  '2. 模型下载\n' +
  '当你从模型市场下载模型时，应用会访问 ModelScope、HuggingFace 或 Modelers 的公开接口。' +
  '如果配置了私有仓库令牌，令牌只保存在本机，仅用于向对应平台发起下载请求。\n\n' +
  '3. 设备权限\n' +
  '麦克风权限仅在你主动录音时使用；相机/相册、文件访问权限仅在你主动添加附件时使用，' +
  '访问范围限于你选择的文件。\n\n' +
  '4. 本地网络服务\n' +
  '本地 API 服务默认只监听 127.0.0.1，由你手动启动和停止；开启局域网监听后，同一网络内的设备可以访问，' +
  '建议同时开启 Bearer 鉴权。\n\n' +
  '5. 数据分析\n' +
  '应用不包含第三方统计、广告或崩溃上报 SDK，不收集设备标识和聊天内容。\n\n' +
  '6. 你的权利\n' +
  '你可以随时在设置中清理缓存、删除会话和模型；卸载应用会删除全部本地数据。\n\n' +
  '继续使用即表示你已阅读并同意以上说明。';

const PRIVACY_POLICY_EN: string =
  'MNN Chat runs entirely on your device, and we respect your privacy.\n\n' +
  '1. Data storage\n' +
  'Chats, attachments, model files and runtime settings stay in the app sandbox on this device and ' +
  'are never uploaded to any server.\n\n' +
  '2. Model downloads\n' +
  'When you download a model from the market, the app calls the public APIs of ModelScope, ' +
  'HuggingFace or Modelers. If you configure a private repository token, it is stored only on this ' +
  'device and used solely for download requests to that platform.\n\n' +
  '3. Device permissions\n' +
  'Microphone access is used only while you record; camera/gallery and file access are used only ' +
  'when you add an attachment, limited to the files you pick.\n\n' +
  '4. Local network service\n' +
  'The local API server listens on 127.0.0.1 by default and is started/stopped by you. When LAN ' +
  'listening is enabled, devices on the same network can reach it; enabling Bearer authentication ' +
  'is recommended.\n\n' +
  '5. Analytics\n' +
  'The app contains no third-party analytics, advertising or crash-reporting SDK and does not ' +
  'collect device identifiers or chat content.\n\n' +
  '6. Your rights\n' +
  'You can clear caches and delete sessions and models in Settings at any time; uninstalling the ' +
  'app removes all local data.\n\n' +
  'By continuing to use the app you confirm that you have read and agree to the above.';

export function privacyPolicyText(): string {
  return I18n.isEnglish() ? PRIVACY_POLICY_EN : PRIVACY_POLICY_ZH;
}