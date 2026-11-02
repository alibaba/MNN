import { common } from '@kit.AbilityKit';
import { BenchmarkRepository } from '../data/BenchmarkRepository';
import { ModelRepository } from '../data/ModelRepository';
import { ModelRuntimeSettings, ModelState, RuntimeBackendType } from '../domain/AppTypes';
import {
  GenerationCoordinator,
  NativeBenchmarkRequest,
  NativeBenchmarkResult,
  NativeBenchmarkSample,
  NativeTarget
} from './GenerationCoordinator';
import { SettingsStore } from './SettingsStore';

export class BenchmarkSummary {
  requestId: string = '';
  modelId: string = '';
  backendType: RuntimeBackendType = RuntimeBackendType.CPU;
  loadMs: number = 0;
  promptTokens: number = 128;
  generateTokens: number = 128;
  repeats: number = 3;
  meanPrefillSpeed: number = 0;
  meanDecodeSpeed: number = 0;
  prefillStdDev: number = 0;
  decodeStdDev: number = 0;
  totalMs: number = 0;
  canceled: boolean = false;
  samples: NativeBenchmarkSample[] = [];
  createdAt: number = 0;
}

export class BenchmarkService {
  private static instance: BenchmarkService = new BenchmarkService();

  static getInstance(): BenchmarkService {
    return BenchmarkService.instance;
  }

  async run(context: common.UIAbilityContext, modelId: string, settingsRevision: number, requestId: string,
    backendType: RuntimeBackendType,
    onProgress: (sample: NativeBenchmarkSample) => void): Promise<BenchmarkSummary> {
    const model = await ModelRepository.getInstance().getModel(modelId);
    if (!model || model.state !== ModelState.READY) {
      throw new Error('Benchmark 模型不可用');
    }
    const settings = await SettingsStore.getInstance().getModelRuntimeSettings(model.id);
    const benchmarkSettings = this.copySettings(settings);
    benchmarkSettings.backendType = backendType;
    benchmarkSettings.promptCache = false;
    const safeModelId = model.id.replace(/[^a-zA-Z0-9._-]/g, '-');
    const runtimeConfigJson = SettingsStore.getInstance().createRuntimeConfigJson(
      benchmarkSettings, `${context.cacheDir}/llm-${safeModelId}`
    );
    const target: NativeTarget = {
      modelId: model.id,
      configPath: model.configPath,
      runtimeConfigJson: runtimeConfigJson
    };
    const loadResult = await GenerationCoordinator.getInstance().prepare(target);
    const request: NativeBenchmarkRequest = {
      requestId: requestId,
      promptTokens: 128,
      generateTokens: 128,
      repeats: 3
    };
    const result = await GenerationCoordinator.getInstance().benchmark(target, request, onProgress);
    const summary = this.summarize(model.id, backendType, loadResult.loadMs, request, result);
    if (!summary.canceled && summary.samples.length > 0) {
      const snapshot = JSON.stringify({
        settingsRevision: settingsRevision,
        runtimeConfig: JSON.parse(runtimeConfigJson)
      });
      await BenchmarkRepository.getInstance().save(model.id, snapshot, JSON.stringify(summary));
    }
    return summary;
  }

  cancel(requestId: string): void {
    GenerationCoordinator.getInstance().cancel(requestId);
  }

  private summarize(modelId: string, backendType: RuntimeBackendType, loadMs: number,
    request: NativeBenchmarkRequest, result: NativeBenchmarkResult): BenchmarkSummary {
    const summary = new BenchmarkSummary();
    summary.requestId = request.requestId;
    summary.modelId = modelId;
    summary.backendType = backendType;
    summary.loadMs = loadMs;
    summary.promptTokens = request.promptTokens;
    summary.generateTokens = request.generateTokens;
    summary.repeats = request.repeats;
    summary.canceled = result.canceled;
    summary.samples = result.samples;
    summary.createdAt = Date.now();
    const prefillSpeeds: number[] = [];
    const decodeSpeeds: number[] = [];
    result.samples.forEach((sample: NativeBenchmarkSample) => {
      if (sample.prefillMs > 0) {
        prefillSpeeds.push(sample.promptTokens * 1000 / sample.prefillMs);
      }
      if (sample.decodeMs > 0) {
        decodeSpeeds.push(sample.generatedTokens * 1000 / sample.decodeMs);
      }
      summary.totalMs += sample.totalMs;
    });
    summary.meanPrefillSpeed = this.mean(prefillSpeeds);
    summary.meanDecodeSpeed = this.mean(decodeSpeeds);
    summary.prefillStdDev = this.sampleStdDev(prefillSpeeds, summary.meanPrefillSpeed);
    summary.decodeStdDev = this.sampleStdDev(decodeSpeeds, summary.meanDecodeSpeed);
    return summary;
  }

  private mean(values: number[]): number {
    if (values.length === 0) {
      return 0;
    }
    let total = 0;
    values.forEach((value: number) => total += value);
    return total / values.length;
  }

  private sampleStdDev(values: number[], mean: number): number {
    if (values.length < 2) {
      return 0;
    }
    let squared = 0;
    values.forEach((value: number) => {
      const difference = value - mean;
      squared += difference * difference;
    });
    return Math.sqrt(squared / (values.length - 1));
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
