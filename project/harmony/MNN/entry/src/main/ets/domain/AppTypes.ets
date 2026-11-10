export enum MessageRole {
  SYSTEM = 'system',
  USER = 'user',
  ASSISTANT = 'assistant'
}

export enum GenerationState {
  COMPLETE = 'complete',
  GENERATING = 'generating',
  CANCELED = 'canceled',
  FAILED = 'failed'
}

export enum ModelState {
  READY = 'ready',
  INVALID = 'invalid',
  DOWNLOADING = 'downloading'
}

export class GenerationMetrics {
  loadMs: number = 0;
  firstTokenMs: number = 0;
  promptTokens: number = 0;
  generatedTokens: number = 0;
  prefillMs: number = 0;
  decodeMs: number = 0;
  totalMs: number = 0;
}

export class ChatSessionRecord {
  id: string = '';
  title: string = '';
  modelId: string = '';
  createdAt: number = 0;
  updatedAt: number = 0;
}

export class ChatMessageRecord {
  id: string = '';
  sessionId: string = '';
  role: MessageRole = MessageRole.USER;
  rawContent: string = '';
  displayContent: string = '';
  reasoningContent: string = '';
  state: GenerationState = GenerationState.COMPLETE;
  createdAt: number = 0;
  metrics: GenerationMetrics = new GenerationMetrics();
}

export class AttachmentRecord {
  id: string = '';
  messageId: string = '';
  mediaType: string = '';
  localPath: string = '';
  displayName: string = '';
  sizeBytes: number = 0;
  createdAt: number = 0;
}

export class ModelRecord {
  id: string = '';
  name: string = '';
  configPath: string = '';
  rootPath: string = '';
  state: ModelState = ModelState.INVALID;
  isLegacy: boolean = false;
  isProtected: boolean = false;
  isPinned: boolean = false;
  sizeBytes: number = 0;
  lastUsedAt: number = 0;
  createdAt: number = 0;
}

export class ModelStorageStats {
  modelBytes: number = 0;
  freeBytes: number = 0;
  totalBytes: number = 0;
}

export class ModelCapabilities {
  vision: boolean = false;
  audioInput: boolean = false;
  videoInput: boolean = false;
  audioOutput: boolean = false;
  diffusion: boolean = false;
  attachmentReason: string = '当前模型不支持多媒体输入';
  videoReason: string = '当前构建未启用视频解码';
  diffusionReason: string = '当前构建未启用扩散运行时';
}

export enum RuntimeBackendType {
  CPU = 'cpu',
  OPENCL = 'opencl'
}

export enum RuntimePrecision {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high'
}

export enum RuntimeMemory {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high'
}

export enum RuntimeSamplerType {
  GREEDY = 'greedy',
  MIXED = 'mixed',
  PENALTY = 'penalty',
  TEMPERATURE = 'temperature',
  TOP_K = 'topK',
  TOP_P = 'topP'
}

// Defaults mirror the Android MnnLlmChat ModelConfig.defaultConfig so the two clients behave the same.
export class ModelRuntimeSettings {
  backendType: RuntimeBackendType = RuntimeBackendType.CPU;
  threadNum: number = 4;
  precision: RuntimePrecision = RuntimePrecision.LOW;
  memory: RuntimeMemory = RuntimeMemory.LOW;
  useMmap: boolean = false;
  promptCache: boolean = false;
  maxTokens: number = 2048;
  systemPrompt: string = 'You are a helpful assistant.';
  thinking: boolean = true;
  samplerType: RuntimeSamplerType = RuntimeSamplerType.MIXED;
  temperature: number = 0.6;
  topP: number = 0.95;
  topK: number = 20;
  minP: number = 0.05;
  tfsZ: number = 1.0;
  typical: number = 0.95;
  penalty: number = 1.02;
  presencePenalty: number = 0.0;
  frequencyPenalty: number = 0.0;
  penaltyWindow: number = 0;
  nGram: number = 8;
  nGramFactor: number = 1.02;
  diffusionSteps: number = 10;
  imageWidth: number = 512;
  imageHeight: number = 512;
  diffusionSeed: number = 42;
  diffusionMemoryMode: number = 2;
}

export class MarketSources {
  HuggingFace?: string;
  ModelScope?: string;
  Modelers?: string;
}

export class MarketModelRecord {
  modelName: string = '';
  vendor: string = '';
  sizeB: number = 0;
  fileSize: number = 0;
  description: string = '';
  tags: string[] = [];
  categories: string[] = [];
  sources: MarketSources = new MarketSources();
}

export class MarketCatalogRecord {
  version: string = '0';
  quickFilterTags: string[] = [];
  vendorOrder: string[] = [];
  models: MarketModelRecord[] = [];
}

export class ProviderFileRecord {
  relativePath: string = '';
  downloadUrl: string = '';
  sizeBytes: number = 0;
  sha256: string = '';
}

export enum DownloadState {
  PENDING = 'pending',
  DOWNLOADING = 'downloading',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export class DownloadTaskRecord {
  id: string = '';
  modelId: string = '';
  provider: string = '';
  sourceUrl: string = '';
  targetPath: string = '';
  state: string = '';
  downloadedBytes: number = 0;
  totalBytes: number = 0;
  updatedAt: number = 0;
}

export class ModelDownloadStatus {
  modelId: string = '';
  state: DownloadState = DownloadState.PENDING;
  downloadedBytes: number = 0;
  totalBytes: number = 0;
  error: string = '';
}

export class BenchmarkRecord {
  id: string = '';
  modelId: string = '';
  scenario: string = '';
  configSnapshot: string = '';
  resultJson: string = '';
  createdAt: number = 0;
}
