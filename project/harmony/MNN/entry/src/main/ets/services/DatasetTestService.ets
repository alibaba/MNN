import { common } from '@kit.AbilityKit';
import { fileIo } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';
import { ModelRepository } from '../data/ModelRepository';
import { ModelRuntimeSettings, ModelState } from '../domain/AppTypes';
import { GenerationCoordinator, NativeGenerateResult, NativeTarget } from './GenerationCoordinator';
import { SettingsStore } from './SettingsStore';

export class DatasetItemResult {
  index: number = 0;
  promptChars: number = 0;
  output: string = '';
  promptTokens: number = 0;
  generatedTokens: number = 0;
  prefillMs: number = 0;
  decodeMs: number = 0;
  error: string = '';
}

export class DatasetTestResult {
  fileName: string = '';
  modelId: string = '';
  items: DatasetItemResult[] = [];
  canceled: boolean = false;
}

interface DatasetLine {
  prompt?: string;
}

const MAX_DATASET_ITEMS: number = 3;
const MAX_ANSWER_TOKENS: number = 256;
const MAX_DATASET_BYTES: number = 32 * 1024 * 1024;

/**
 * Runs a JSONL prompt dataset (for example NeedleBench) through the active model, mirroring the
 * Android client's dataset benchmark. Each line must be an object with a "prompt" field.
 */
export class DatasetTestService {
  private static instance: DatasetTestService = new DatasetTestService();
  private requestId: string = '';

  static getInstance(): DatasetTestService {
    return DatasetTestService.instance;
  }

  readPrompts(uri: string): string[] {
    // DocumentViewPicker returns a docs URI, so read through a file descriptor rather than a path.
    const file = fileIo.openSync(uri, fileIo.OpenMode.READ_ONLY);
    let text = '';
    try {
      const size = fileIo.statSync(file.fd).size;
      if (size > MAX_DATASET_BYTES) {
        throw new Error('数据集文件过大（最多 32MB）');
      }
      const buffer = new ArrayBuffer(size);
      const read = fileIo.readSync(file.fd, buffer);
      text = util.TextDecoder.create('utf-8').decodeToString(new Uint8Array(buffer, 0, read));
    } finally {
      fileIo.closeSync(file);
    }
    const prompts: string[] = [];
    text.split('\n').forEach((line: string) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return;
      }
      try {
        const item = JSON.parse(trimmed) as DatasetLine;
        if (item.prompt && item.prompt.length > 0) {
          prompts.push(item.prompt);
        }
      } catch (error) {
        // Skip malformed lines instead of aborting the whole dataset.
      }
    });
    if (prompts.length === 0) {
      throw new Error('数据集中没有可用的 prompt');
    }
    return prompts;
  }

  cancel(): void {
    if (this.requestId) {
      GenerationCoordinator.getInstance().cancel(this.requestId);
    }
  }

  async run(context: common.UIAbilityContext, modelId: string, fileName: string, prompts: string[],
    onProgress: (index: number, total: number) => void): Promise<DatasetTestResult> {
    const result = new DatasetTestResult();
    result.fileName = fileName;
    result.modelId = modelId;
    const model = await ModelRepository.getInstance().getModel(modelId);
    if (!model || model.state !== ModelState.READY) {
      throw new Error('数据集测试模型不可用');
    }
    const settings = await SettingsStore.getInstance().getModelRuntimeSettings(model.id);
    const testSettings = this.copySettings(settings);
    testSettings.promptCache = false;
    const safeModelId = model.id.replace(/[^a-zA-Z0-9._-]/g, '-');
    const target: NativeTarget = {
      modelId: model.id,
      configPath: model.configPath,
      runtimeConfigJson: SettingsStore.getInstance().createRuntimeConfigJson(
        testSettings, `${context.cacheDir}/llm-${safeModelId}`)
    };
    await GenerationCoordinator.getInstance().prepare(target);
    this.requestId = GenerationCoordinator.getInstance().createRequestId('dataset');
    const coordinator = GenerationCoordinator.getInstance();
    const limit = Math.min(prompts.length, MAX_DATASET_ITEMS);
    for (let index = 0; index < limit; index++) {
      onProgress(index + 1, limit);
      const item = new DatasetItemResult();
      item.index = index + 1;
      item.promptChars = prompts[index].length;
      try {
        const generated: NativeGenerateResult = await coordinator.generate(target, {
          requestId: `${this.requestId}:${index}`,
          messages: [{ role: 'user', content: prompts[index] }],
          maxTokens: Math.min(MAX_ANSWER_TOKENS, settings.maxTokens)
        }, () => {
        });
        item.output = generated.text;
        item.promptTokens = generated.promptTokens;
        item.generatedTokens = generated.generatedTokens;
        item.prefillMs = generated.prefillMs;
        item.decodeMs = generated.decodeMs;
        if (generated.canceled) {
          result.canceled = true;
        }
      } catch (error) {
        item.error = (error as Error).message;
      }
      result.items.push(item);
      if (result.canceled) {
        break;
      }
    }
    this.requestId = '';
    return result;
  }

  private copySettings(settings: ModelRuntimeSettings): ModelRuntimeSettings {
    const copy = new ModelRuntimeSettings();
    copy.backendType = settings.backendType;
    copy.threadNum = settings.threadNum;
    copy.precision = settings.precision;
    copy.memory = settings.memory;
    copy.useMmap = settings.useMmap;
    copy.promptCache = settings.promptCache;
    copy.maxTokens = settings.maxTokens;
    copy.systemPrompt = settings.systemPrompt;
    copy.thinking = settings.thinking;
    copy.samplerType = settings.samplerType;
    copy.temperature = settings.temperature;
    copy.topP = settings.topP;
    copy.topK = settings.topK;
    return copy;
  }
}