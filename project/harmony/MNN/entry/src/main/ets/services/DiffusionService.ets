import { common } from '@kit.AbilityKit';
import fs from '@ohos.file.fs';
import testNapi, { LlmDiffusionRequest } from 'libentry.so';
import { ModelRecord } from '../domain/AppTypes';

interface DiffusionConfigDocument {
  model_type?: string;
  backend_type?: string;
  diffusion_steps?: number;
  image_width?: number;
  image_height?: number;
  diffusion_seed?: number;
  diffusion_memory_mode?: string;
}

export class DiffusionResult {
  outputPath: string = '';
  totalMs: number = 0;
}

/**
 * Wraps the native diffusion engine (Sana / SD) so the chat page can generate images.
 * Parameters come from the model's config.json, mirroring the Android diffusion defaults.
 */
export class DiffusionService {
  private static instance: DiffusionService = new DiffusionService();

  static getInstance(): DiffusionService {
    return DiffusionService.instance;
  }

  isDiffusionModel(model: ModelRecord): boolean {
    const name = `${model.name} ${model.id}`.toLowerCase();
    return name.indexOf('stable-diffusion') >= 0 || name.indexOf('sana') >= 0;
  }

  async generate(context: common.UIAbilityContext, model: ModelRecord, sessionId: string, prompt: string,
    imageInput: string, onProgress: (progress: number) => void): Promise<DiffusionResult> {
    const config = this.readConfig(model.configPath);
    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fs.accessSync(directory)) {
      fs.mkdirSync(directory, true);
    }
    const outputPath = `${directory}/image-${Date.now()}.png`;
    const seed = config.diffusion_seed === undefined || config.diffusion_seed < 0 ?
      Math.floor(Math.random() * 2147483647) : config.diffusion_seed;
    const request: LlmDiffusionRequest = {
      requestId: `diffusion:${Date.now()}`,
      resourcePath: model.rootPath,
      prompt: prompt,
      outputPath: outputPath,
      modelType: `${model.name} ${model.id}`.toLowerCase().indexOf('sana') >= 0 ? 2 : 0,
      backendType: (config.backend_type || 'cpu').toLowerCase() === 'opencl' ? 3 : 0,
      memoryMode: Number(config.diffusion_memory_mode || '0'),
      width: config.image_width || 512,
      height: config.image_height || 512,
      steps: config.diffusion_steps || 10,
      seed: seed,
      imageInput: imageInput,
      negativePrompt: '',
      useCfg: true,
      cfgScale: 4.5
    };
    const nativeResult = await testNapi.runDiffusion(request, onProgress);
    const result = new DiffusionResult();
    result.outputPath = outputPath;
    result.totalMs = nativeResult.totalMs;
    return result;
  }

  private readConfig(configPath: string): DiffusionConfigDocument {
    try {
      return JSON.parse(fs.readTextSync(configPath)) as DiffusionConfigDocument;
    } catch (error) {
      return {};
    }
  }

  private safeName(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, '-').substring(0, 96);
  }
}