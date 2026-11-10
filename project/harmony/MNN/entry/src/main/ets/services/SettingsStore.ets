import { common } from '@kit.AbilityKit';
import preferences from '@ohos.data.preferences';
import fs from '@ohos.file.fs';
import {
  ModelRuntimeSettings,
  RuntimeBackendType,
  RuntimeMemory,
  RuntimePrecision,
  RuntimeSamplerType
} from '../domain/AppTypes';

const SETTINGS_NAME: string = 'mnn_settings';
const MODEL_SETTINGS_PREFIX: string = 'model_runtime_settings.';
export const ACTIVE_MODEL_ID_KEY: string = 'active_model_id';
export const SETTINGS_REVISION_KEY: string = 'model_runtime_settings_revision';
export const APP_SHOW_METRICS_KEY: string = 'app_show_metrics';
export const APP_STOP_DOWNLOAD_ON_CHAT_KEY: string = 'app_stop_download_on_chat';
export const APP_AUDIO_REPLY_KEY: string = 'app_audio_reply';
export const APP_VIDEO_FRAMES_KEY: string = 'app_video_frames';
export const APP_VIDEO_FRAME_SIZE_KEY: string = 'app_video_frame_size';

interface RuntimeSettingsDocument {
  backend_type?: string;
  thread_num?: number;
  precision?: string;
  memory?: string;
  use_mmap?: boolean;
  prompt_cache?: boolean;
  max_new_tokens?: number;
  system_prompt?: string;
  jinja?: RuntimeJinja;
  sampler_type?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  tfs_z?: number;
  typical?: number;
  repetition_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  penalty_window?: number;
  n_gram?: number;
  ngram_factor?: number;
  diffusion_steps?: number;
  image_width?: number;
  image_height?: number;
  diffusion_seed?: number;
  diffusion_memory_mode?: number;
}

interface RuntimeJinjaContext {
  enable_thinking: boolean;
}

interface RuntimeJinja {
  context: RuntimeJinjaContext;
}

interface NativeRuntimeConfigDocument {
  tmp_path: string;
  async: boolean;
  backend_type: string;
  thread_num: number;
  precision: string;
  memory: string;
  use_mmap: boolean;
  prompt_cache: boolean;
  sampler_type: string;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  tfs_z: number;
  typical: number;
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  penalty_window: number;
  n_gram: number;
  ngram_factor: number;
  max_new_tokens: number;
  system_prompt: string;
  jinja: RuntimeJinja;
}

export class SettingsStore {
  private static instance: SettingsStore = new SettingsStore();
  private store?: preferences.Preferences;
  private filesDir: string = '';
  private revision: number = 0;

  static getInstance(): SettingsStore {
    return SettingsStore.instance;
  }

  async initialize(context: common.UIAbilityContext): Promise<void> {
    if (!this.store) {
      this.store = await preferences.getPreferences(context, SETTINGS_NAME);
      this.filesDir = context.filesDir;
      const revision = await this.store.get(SETTINGS_REVISION_KEY, 0);
      this.revision = revision as number;
    }
  }

  async getString(key: string, defaultValue: string = ''): Promise<string> {
    const value = await this.getStore().get(key, defaultValue);
    return value as string;
  }

  async putString(key: string, value: string): Promise<void> {
    const store = this.getStore();
    await store.put(key, value);
    await store.flush();
  }

  async getNumber(key: string, defaultValue: number): Promise<number> {
    const value = await this.getStore().get(key, defaultValue);
    return value as number;
  }

  async putNumber(key: string, value: number): Promise<void> {
    const store = this.getStore();
    await store.put(key, value);
    await store.flush();
  }

  async getBoolean(key: string, defaultValue: boolean): Promise<boolean> {
    const value = await this.getStore().get(key, defaultValue);
    return value as boolean;
  }

  async putBoolean(key: string, value: boolean): Promise<void> {
    const store = this.getStore();
    await store.put(key, value);
    await store.flush();
  }

  async getModelRuntimeSettings(modelId: string): Promise<ModelRuntimeSettings> {
    const key = `${MODEL_SETTINGS_PREFIX}${modelId}`;
    let value = await this.getString(key);
    const configPath = this.modelConfigPath(modelId);
    if (!value && fs.accessSync(configPath)) {
      value = fs.readTextSync(configPath);
    }
    if (!value) {
      return new ModelRuntimeSettings();
    }
    try {
      return this.hydrateSettings(JSON.parse(value) as RuntimeSettingsDocument);
    } catch (error) {
      return new ModelRuntimeSettings();
    }
  }

  async putModelRuntimeSettings(modelId: string, settings: ModelRuntimeSettings): Promise<number> {
    const document = this.toDocument(settings);
    const value = JSON.stringify(document);
    const configDirectory = this.modelConfigDirectory(modelId);
    if (!fs.accessSync(configDirectory)) {
      fs.mkdirSync(configDirectory, true);
    }
    const file = fs.openSync(this.modelConfigPath(modelId),
      fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
    try {
      fs.writeSync(file.fd, value);
    } finally {
      fs.closeSync(file);
    }

    const store = this.getStore();
    const revision = ++this.revision;
    await store.put(`${MODEL_SETTINGS_PREFIX}${modelId}`, value);
    await store.put(SETTINGS_REVISION_KEY, revision);
    await store.flush();
    return revision;
  }

  async resetModelRuntimeSettings(modelId: string): Promise<number> {
    const store = this.getStore();
    await store.delete(`${MODEL_SETTINGS_PREFIX}${modelId}`);
    const configPath = this.modelConfigPath(modelId);
    if (fs.accessSync(configPath)) {
      fs.unlinkSync(configPath);
    }
    const revision = ++this.revision;
    await store.put(SETTINGS_REVISION_KEY, revision);
    await store.flush();
    return revision;
  }

  async getSettingsRevision(): Promise<number> {
    return this.revision;
  }

  createRuntimeConfigJson(settings: ModelRuntimeSettings, tmpPath: string): string {
    const document: NativeRuntimeConfigDocument = {
      tmp_path: tmpPath,
      async: false,
      backend_type: settings.backendType,
      thread_num: settings.threadNum,
      precision: settings.precision,
      memory: settings.memory,
      use_mmap: settings.useMmap,
      prompt_cache: settings.promptCache,
      sampler_type: settings.samplerType,
      temperature: settings.temperature,
      top_p: settings.topP,
      top_k: settings.topK,
      min_p: settings.minP,
      tfs_z: settings.tfsZ,
      typical: settings.typical,
      repetition_penalty: settings.penalty,
      presence_penalty: settings.presencePenalty,
      frequency_penalty: settings.frequencyPenalty,
      penalty_window: settings.penaltyWindow,
      n_gram: settings.nGram,
      ngram_factor: settings.nGramFactor,
      max_new_tokens: settings.maxTokens,
      system_prompt: settings.systemPrompt,
      jinja: { context: { enable_thinking: settings.thinking } }
    };
    return JSON.stringify(document);
  }

  private hydrateSettings(document: RuntimeSettingsDocument): ModelRuntimeSettings {
    const settings = new ModelRuntimeSettings();
    settings.backendType = document.backend_type === RuntimeBackendType.OPENCL ? RuntimeBackendType.OPENCL :
      RuntimeBackendType.CPU;
    settings.threadNum = this.clamp(document.thread_num, 1, 16, settings.threadNum);
    settings.precision = this.parsePrecision(document.precision);
    settings.memory = this.parseMemory(document.memory);
    settings.useMmap = document.use_mmap === undefined ? settings.useMmap : document.use_mmap;
    settings.promptCache = document.prompt_cache === undefined ? settings.promptCache : document.prompt_cache;
    settings.maxTokens = this.clamp(document.max_new_tokens, 1, 4096, settings.maxTokens);
    settings.systemPrompt = document.system_prompt || settings.systemPrompt;
    settings.thinking = document.jinja?.context.enable_thinking === undefined ? settings.thinking :
      document.jinja.context.enable_thinking;
    settings.samplerType = this.parseSampler(document.sampler_type);
    settings.temperature = this.clamp(document.temperature, 0.05, 2, settings.temperature);
    settings.topP = this.clamp(document.top_p, 0, 1, settings.topP);
    settings.topK = this.clamp(document.top_k, 1, 200, settings.topK);
    settings.minP = this.clamp(document.min_p, 0, 1, settings.minP);
    settings.tfsZ = this.clamp(document.tfs_z, 0, 1, settings.tfsZ);
    settings.typical = this.clamp(document.typical, 0, 1, settings.typical);
    settings.penalty = this.clamp(document.repetition_penalty, 1, 2, settings.penalty);
    settings.presencePenalty = this.clamp(document.presence_penalty, 0, 2, settings.presencePenalty);
    settings.frequencyPenalty = this.clamp(document.frequency_penalty, 0, 2, settings.frequencyPenalty);
    settings.penaltyWindow = this.clamp(document.penalty_window, 0, 1024, settings.penaltyWindow);
    settings.nGram = this.clamp(document.n_gram, 0, 16, settings.nGram);
    settings.nGramFactor = this.clamp(document.ngram_factor, 0, 2, settings.nGramFactor);
    settings.diffusionSteps = this.clamp(document.diffusion_steps, 1, 50, settings.diffusionSteps);
    settings.imageWidth = this.clamp(document.image_width, 256, 1024, settings.imageWidth);
    settings.imageHeight = this.clamp(document.image_height, 256, 1024, settings.imageHeight);
    settings.diffusionSeed = document.diffusion_seed === undefined ? settings.diffusionSeed : document.diffusion_seed;
    settings.diffusionMemoryMode = this.clamp(document.diffusion_memory_mode, 0, 2, settings.diffusionMemoryMode);
    return settings;
  }

  private toDocument(settings: ModelRuntimeSettings): RuntimeSettingsDocument {
    const document: RuntimeSettingsDocument = {
      backend_type: settings.backendType,
      thread_num: this.clamp(settings.threadNum, 1, 16, 4),
      precision: settings.precision,
      memory: settings.memory,
      use_mmap: settings.useMmap,
      prompt_cache: settings.promptCache,
      max_new_tokens: this.clamp(settings.maxTokens, 1, 4096, 256),
      system_prompt: settings.systemPrompt.trim() || 'You are a helpful assistant.',
      jinja: { context: { enable_thinking: settings.thinking } },
      sampler_type: settings.samplerType,
      temperature: this.clamp(settings.temperature, 0.05, 2, 0.6),
      top_p: this.clamp(settings.topP, 0, 1, 0.95),
      top_k: this.clamp(settings.topK, 1, 200, 20),
      min_p: this.clamp(settings.minP, 0, 1, 0.05),
      tfs_z: this.clamp(settings.tfsZ, 0, 1, 1.0),
      typical: this.clamp(settings.typical, 0, 1, 0.95),
      repetition_penalty: this.clamp(settings.penalty, 1, 2, 1.02),
      presence_penalty: this.clamp(settings.presencePenalty, 0, 2, 0),
      frequency_penalty: this.clamp(settings.frequencyPenalty, 0, 2, 0),
      penalty_window: this.clamp(settings.penaltyWindow, 0, 1024, 0),
      n_gram: this.clamp(settings.nGram, 0, 16, 8),
      ngram_factor: this.clamp(settings.nGramFactor, 0, 2, 1.02),
      diffusion_steps: this.clamp(settings.diffusionSteps, 1, 50, 10),
      image_width: this.clamp(settings.imageWidth, 256, 1024, 512),
      image_height: this.clamp(settings.imageHeight, 256, 1024, 512),
      diffusion_seed: settings.diffusionSeed,
      diffusion_memory_mode: this.clamp(settings.diffusionMemoryMode, 0, 2, 2)
    };
    return document;
  }

  private parsePrecision(value?: string): RuntimePrecision {
    if (value === RuntimePrecision.NORMAL) {
      return RuntimePrecision.NORMAL;
    }
    if (value === RuntimePrecision.HIGH) {
      return RuntimePrecision.HIGH;
    }
    return RuntimePrecision.LOW;
  }

  private parseMemory(value?: string): RuntimeMemory {
    if (value === RuntimeMemory.NORMAL) {
      return RuntimeMemory.NORMAL;
    }
    if (value === RuntimeMemory.HIGH) {
      return RuntimeMemory.HIGH;
    }
    return RuntimeMemory.LOW;
  }

  private parseSampler(value?: string): RuntimeSamplerType {
    if (value === RuntimeSamplerType.GREEDY) {
      return RuntimeSamplerType.GREEDY;
    }
    if (value === RuntimeSamplerType.PENALTY) {
      return RuntimeSamplerType.PENALTY;
    }
    if (value === RuntimeSamplerType.TEMPERATURE) {
      return RuntimeSamplerType.TEMPERATURE;
    }
    if (value === RuntimeSamplerType.TOP_K) {
      return RuntimeSamplerType.TOP_K;
    }
    if (value === RuntimeSamplerType.TOP_P) {
      return RuntimeSamplerType.TOP_P;
    }
    return RuntimeSamplerType.MIXED;
  }

  private clamp(value: number | undefined, minimum: number, maximum: number, fallback: number): number {
    if (value === undefined || !Number.isFinite(value)) {
      return fallback;
    }
    return Math.min(maximum, Math.max(minimum, value));
  }

  private modelConfigDirectory(modelId: string): string {
    return `${this.filesDir}/configs/${this.safeModelId(modelId)}`;
  }

  private modelConfigPath(modelId: string): string {
    return `${this.modelConfigDirectory(modelId)}/custom_config.json`;
  }

  private safeModelId(modelId: string): string {
    return modelId.replace(/[^a-zA-Z0-9._-]/g, '-').substring(0, 96);
  }

  private getStore(): preferences.Preferences {
    if (!this.store) {
      throw new Error('应用设置尚未初始化');
    }
    return this.store;
  }
}
