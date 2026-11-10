import { common, Want } from '@kit.AbilityKit';
import bundleManager from '@ohos.bundle.bundleManager';
import fs from '@ohos.file.fs';
import { ModelRepository } from '../../data/ModelRepository';
import {
  ModelRuntimeSettings,
  RuntimeBackendType,
  RuntimeMemory,
  RuntimePrecision,
  RuntimeSamplerType
} from '../../domain/AppTypes';
import { mnnVersion } from '../../services/GenerationCoordinator';
import { MarketAuth } from '../../services/MarketAuth';
import {
  PROVIDER_HUGGING_FACE,
  PROVIDER_MODEL_SCOPE,
  PROVIDER_MODELERS
} from '../../services/ProviderAdapters';
import {
  APP_AUDIO_REPLY_KEY,
  APP_SHOW_METRICS_KEY,
  APP_STOP_DOWNLOAD_ON_CHAT_KEY,
  APP_VIDEO_FRAMES_KEY,
  APP_VIDEO_FRAME_SIZE_KEY,
  SettingsStore
} from '../../services/SettingsStore';

class MarketTokenForm {
  huggingFace: string = '';
  modelScope: string = '';
  modelers: string = '';
}

@Component
export struct SettingsPage {
  @Watch('onActiveModelChanged') @Link activeModelId: string;
  @Link settingsRevision: number;
  @State private modelName: string = '';
  @State private settings: ModelRuntimeSettings = new ModelRuntimeSettings();
  @State private status: string = '';
  @State private error: string = '';
  @State private busy: boolean = false;
  @State private appVersion: string = '';
  @State private engineVersion: string = '';
  @State private cacheBytes: number = 0;
  @StorageLink('showMetrics') private showMetrics: boolean = true;
  @StorageLink('audioReply') private audioReply: boolean = false;
  @State private stopDownloadsOnChat: boolean = true;
  @State private videoFrames: number = 4;
  @State private videoFrameSize: number = 320;
  @State private isDiffusion: boolean = false;
  @State private tokens: MarketTokenForm = new MarketTokenForm();
  private loadSequence: number = 0;

  aboutToAppear(): void {
    this.loadSettings();
    this.refreshAppInfo();
    SettingsStore.getInstance().getBoolean(APP_SHOW_METRICS_KEY, true).then((value: boolean) => {
      this.showMetrics = value;
    }).catch((error: Error) => {
      this.error = error.message;
    });
    SettingsStore.getInstance().getBoolean(APP_STOP_DOWNLOAD_ON_CHAT_KEY, true).then((value: boolean) => {
      this.stopDownloadsOnChat = value;
    }).catch((error: Error) => {
      this.error = error.message;
    });
    SettingsStore.getInstance().getBoolean(APP_AUDIO_REPLY_KEY, false).then((value: boolean) => {
      this.audioReply = value;
    }).catch((error: Error) => {
      this.error = error.message;
    });
    SettingsStore.getInstance().getNumber(APP_VIDEO_FRAMES_KEY, 4).then((value: number) => {
      this.videoFrames = value;
    }).catch((error: Error) => {
      this.error = error.message;
    });
    SettingsStore.getInstance().getNumber(APP_VIDEO_FRAME_SIZE_KEY, 320).then((value: number) => {
      this.videoFrameSize = value;
    }).catch((error: Error) => {
      this.error = error.message;
    });
    this.loadTokens();
  }

  private copyTokens(): MarketTokenForm {
    const next = new MarketTokenForm();
    next.huggingFace = this.tokens.huggingFace;
    next.modelScope = this.tokens.modelScope;
    next.modelers = this.tokens.modelers;
    return next;
  }

  private loadTokens(): void {
    Promise.all([
      MarketAuth.getToken(PROVIDER_HUGGING_FACE),
      MarketAuth.getToken(PROVIDER_MODEL_SCOPE),
      MarketAuth.getToken(PROVIDER_MODELERS)
    ]).then((values: string[]) => {
      const next = new MarketTokenForm();
      next.huggingFace = values[0];
      next.modelScope = values[1];
      next.modelers = values[2];
      this.tokens = next;
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private saveTokens(): void {
    Promise.all([
      MarketAuth.saveToken(PROVIDER_HUGGING_FACE, this.tokens.huggingFace),
      MarketAuth.saveToken(PROVIDER_MODEL_SCOPE, this.tokens.modelScope),
      MarketAuth.saveToken(PROVIDER_MODELERS, this.tokens.modelers)
    ]).then(() => {
      this.status = '模型市场令牌已保存';
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private openWebPage(url: string): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    try {
      const want: Want = {
        action: 'ohos.want.action.viewData',
        entities: ['entity.system.browsable'],
        uri: url
      };
      context.startAbility(want).catch((error: Error) => {
        this.error = `无法打开浏览器：${error.message}`;
      });
    } catch (error) {
      this.error = `无法打开浏览器：${(error as Error).message}`;
    }
  }

  private checkForUpdate(): void {
    const version = this.appVersion ? `当前版本 ${this.appVersion}\n\n` : '';
    AlertDialog.show({
      title: '检查更新',
      message: `${version}鸿蒙客户端随 MNN 仓库一起发布，请到 GitHub Releases 查看最新版本与更新说明。`,
      primaryButton: {
        value: '打开 Releases',
        action: () => this.openWebPage('https://github.com/alibaba/MNN/releases')
      },
      secondaryButton: {
        value: '关闭',
        action: () => {}
      }
    });
  }

  private showAbout(): void {
    const version = this.appVersion ? `版本 ${this.appVersion}\n\n` : '';
    const engine = this.engineVersion;
    AlertDialog.show({
      title: '关于 MNN Chat',
      message: `${version}${engine ? `MNN 引擎 ${engine}\n\n` : ''}基于 MNN 推理引擎的端侧大模型聊天客户端。\n\n` +
        '聊天记录、模型与配置仅保存在本机，推理过程完全离线。',
      primaryButton: {
        value: '项目主页',
        action: () => this.openWebPage('https://github.com/alibaba/MNN')
      },
      secondaryButton: {
        value: '关闭',
        action: () => {}
      }
    });
  }

  private toggleAudioReply(value: boolean): void {
    this.audioReply = value;
    SettingsStore.getInstance().putBoolean(APP_AUDIO_REPLY_KEY, value).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private toggleStopDownloadsOnChat(value: boolean): void {
    this.stopDownloadsOnChat = value;
    SettingsStore.getInstance().putBoolean(APP_STOP_DOWNLOAD_ON_CHAT_KEY, value).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private toggleShowMetrics(value: boolean): void {
    this.showMetrics = value;
    SettingsStore.getInstance().putBoolean(APP_SHOW_METRICS_KEY, value).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private refreshAppInfo(): void {
    try {
      const info = bundleManager.getBundleInfoForSelfSync(bundleManager.BundleFlag.GET_BUNDLE_INFO_DEFAULT);
      this.appVersion = info.versionName;
    } catch (error) {
      this.appVersion = '';
    }
    this.engineVersion = mnnVersion();
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.cacheBytes = this.directorySize(context.cacheDir);
  }

  private directorySize(path: string): number {
    try {
      let total = 0;
      const names = fs.listFileSync(path);
      for (let index = 0; index < names.length; index++) {
        const child = `${path}/${names[index]}`;
        const stat = fs.statSync(child);
        total += stat.isDirectory() ? this.directorySize(child) : stat.size;
      }
      return total;
    } catch (error) {
      return 0;
    }
  }

  private clearCache(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    try {
      const names = fs.listFileSync(context.cacheDir);
      for (let index = 0; index < names.length; index++) {
        const child = `${context.cacheDir}/${names[index]}`;
        const stat = fs.statSync(child);
        if (stat.isDirectory()) {
          fs.rmdirSync(child);
        } else {
          fs.unlinkSync(child);
        }
      }
      this.cacheBytes = 0;
      this.status = '缓存已清理';
      this.error = '';
    } catch (error) {
      this.error = (error as Error).message;
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  private onActiveModelChanged(): void {
    this.loadSettings();
  }

  private loadSettings(): void {
    const modelId = this.activeModelId;
    if (!modelId) {
      return;
    }
    const sequence = ++this.loadSequence;
    ModelRepository.getInstance().getModel(modelId).then(async (model) => {
      const modelName = model?.name || modelId;
      const capabilities = await ModelRepository.getInstance().getCapabilities(modelId);
      const settings = await SettingsStore.getInstance().getModelRuntimeSettings(modelId);
      if (sequence !== this.loadSequence || modelId !== this.activeModelId) {
        return;
      }
      this.modelName = modelName;
      this.isDiffusion = capabilities.diffusion;
      this.settings = settings;
      this.status = '';
      this.error = '';
    }).catch((error: Error) => {
      if (sequence === this.loadSequence) {
        this.error = error.message;
      }
    });
  }

  private copySettings(): ModelRuntimeSettings {
    const copy = new ModelRuntimeSettings();
    copy.backendType = this.settings.backendType;
    copy.threadNum = this.settings.threadNum;
    copy.precision = this.settings.precision;
    copy.memory = this.settings.memory;
    copy.useMmap = this.settings.useMmap;
    copy.promptCache = this.settings.promptCache;
    copy.maxTokens = this.settings.maxTokens;
    copy.systemPrompt = this.settings.systemPrompt;
    copy.thinking = this.settings.thinking;
    copy.samplerType = this.settings.samplerType;
    copy.temperature = this.settings.temperature;
    copy.topP = this.settings.topP;
    copy.topK = this.settings.topK;
    copy.minP = this.settings.minP;
    copy.tfsZ = this.settings.tfsZ;
    copy.typical = this.settings.typical;
    copy.penalty = this.settings.penalty;
    copy.presencePenalty = this.settings.presencePenalty;
    copy.frequencyPenalty = this.settings.frequencyPenalty;
    copy.penaltyWindow = this.settings.penaltyWindow;
    copy.nGram = this.settings.nGram;
    copy.nGramFactor = this.settings.nGramFactor;
    copy.diffusionSteps = this.settings.diffusionSteps;
    copy.imageWidth = this.settings.imageWidth;
    copy.imageHeight = this.settings.imageHeight;
    copy.diffusionSeed = this.settings.diffusionSeed;
    copy.diffusionMemoryMode = this.settings.diffusionMemoryMode;
    return copy;
  }

  private samplerIndex(): number {
    switch (this.settings.samplerType) {
      case RuntimeSamplerType.MIXED:
        return 0;
      case RuntimeSamplerType.GREEDY:
        return 1;
      case RuntimeSamplerType.PENALTY:
        return 2;
      case RuntimeSamplerType.TEMPERATURE:
        return 3;
      case RuntimeSamplerType.TOP_K:
        return 4;
      default:
        return 5;
    }
  }

  private save(): void {
    if (this.busy) {
      return;
    }
    this.busy = true;
    SettingsStore.getInstance().putModelRuntimeSettings(this.activeModelId, this.settings).then((revision: number) => {
      this.settingsRevision = revision;
      this.status = '设置已保存，将在下次生成前应用';
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private reset(): void {
    if (this.busy) {
      return;
    }
    this.busy = true;
    SettingsStore.getInstance().resetModelRuntimeSettings(this.activeModelId).then((revision: number) => {
      this.settingsRevision = revision;
      this.loadSettings();
      this.status = '已恢复默认设置';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private videoFrameIndex(value: number): number {
    const values: number[] = [2, 4, 6, 8];
    const index = values.indexOf(value);
    return index >= 0 ? index : 1;
  }

  private videoFrameAt(index: number): number {
    const values: number[] = [2, 4, 6, 8];
    return index >= 0 && index < values.length ? values[index] : 4;
  }

  private videoSizeIndex(value: number): number {
    const values: number[] = [256, 320, 384];
    const index = values.indexOf(value);
    return index >= 0 ? index : 1;
  }

  private videoSizeAt(index: number): number {
    const values: number[] = [256, 320, 384];
    return index >= 0 && index < values.length ? values[index] : 320;
  }

  private updateVideoFrames(value: number): void {
    this.videoFrames = value;
    SettingsStore.getInstance().putNumber(APP_VIDEO_FRAMES_KEY, value).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private updateVideoFrameSize(value: number): void {
    this.videoFrameSize = value;
    SettingsStore.getInstance().putNumber(APP_VIDEO_FRAME_SIZE_KEY, value).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private stepIndex(steps: number): number {
    const values: number[] = [1, 2, 4, 6, 8, 10, 12, 16, 20];
    const index = values.indexOf(steps);
    return index >= 0 ? index : 5;
  }

  private stepAt(index: number): number {
    const values: number[] = [1, 2, 4, 6, 8, 10, 12, 16, 20];
    return index >= 0 && index < values.length ? values[index] : 10;
  }

  private imageSizeIndex(size: number): number {
    const sizes: number[] = [256, 384, 512, 640, 768, 1024];
    const index = sizes.indexOf(size);
    return index >= 0 ? index : 2;
  }

  private imageSizeAt(index: number): number {
    const sizes: number[] = [256, 384, 512, 640, 768, 1024];
    return index >= 0 && index < sizes.length ? sizes[index] : 512;
  }

  private memoryModeLabel(mode: number): string {
    if (mode === 0) {
      return '省内存';
    }
    return mode === 1 ? '高性能' : '均衡';
  }

  private memoryModeIndex(mode: number): number {
    if (mode === 0) {
      return 0;
    }
    return mode === 1 ? 2 : 1;
  }

  @Builder
  private settingTitle(title: string, value: string) {
    Row() {
      Text(title)
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .layoutWeight(1)
      Text(value)
        .fontSize(13)
        .fontColor('#6B7280')
    }
    .width('100%')
  }

  build() {
    Column() {
      Column({ space: 4 }) {
        Text('推理设置')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .width('100%')
        Text(this.modelName || this.activeModelId)
          .fontSize(13)
          .fontColor('#6B7280')
          .width('100%')
      }
      .padding({ left: 20, right: 20, top: 18, bottom: 12 })

      Scroll() {
        Column({ space: 16 }) {
          if (this.isDiffusion) {
            Column({ space: 8 }) {
              Row() {
                Text('生成步数')
                  .fontSize(15)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(`${this.settings.diffusionSteps} 步`)
                  .fontSize(13)
                  .fontColor('#6B7280')
              }
              .width('100%')
              Select([
                { value: '1' },
                { value: '2' },
                { value: '4' },
                { value: '6' },
                { value: '8' },
                { value: '10' },
                { value: '12' },
                { value: '16' },
                { value: '20' }
              ])
                .selected(this.stepIndex(this.settings.diffusionSteps))
                .onSelect((index: number) => {
                  const next = this.copySettings();
                  next.diffusionSteps = this.stepAt(index);
                  this.settings = next;
                })
              Text('步数越高越慢；512×512、10 步在 CPU 上约 4 分钟，4 步约 2 分钟')
                .fontSize(11)
                .fontColor('#7A7F87')
                .width('100%')
            }
            .width('100%')
            .padding(14)
            .backgroundColor(Color.White)
            .borderRadius(12)

            Column({ space: 8 }) {
              Row() {
                Text('输出宽度')
                  .fontSize(15)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(`${this.settings.imageWidth}`)
                  .fontSize(13)
                  .fontColor('#6B7280')
              }
              .width('100%')
              Select([
                { value: '256' },
                { value: '384' },
                { value: '512' },
                { value: '640' },
                { value: '768' },
                { value: '1024' }
              ])
                .selected(this.imageSizeIndex(this.settings.imageWidth))
                .onSelect((index: number) => {
                  const next = this.copySettings();
                  next.imageWidth = this.imageSizeAt(index);
                  this.settings = next;
                })
              Row() {
                Text('输出高度')
                  .fontSize(15)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(`${this.settings.imageHeight}`)
                  .fontSize(13)
                  .fontColor('#6B7280')
              }
              .width('100%')
              Select([
                { value: '256' },
                { value: '384' },
                { value: '512' },
                { value: '640' },
                { value: '768' },
                { value: '1024' }
              ])
                .selected(this.imageSizeIndex(this.settings.imageHeight))
                .onSelect((index: number) => {
                  const next = this.copySettings();
                  next.imageHeight = this.imageSizeAt(index);
                  this.settings = next;
                })
            }
            .width('100%')
            .padding(14)
            .backgroundColor(Color.White)
            .borderRadius(12)

            Column({ space: 8 }) {
              Row() {
                Text('随机种子')
                  .fontSize(15)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(this.settings.diffusionSeed < 0 ? '随机' : `${this.settings.diffusionSeed}`)
                  .fontSize(13)
                  .fontColor('#6B7280')
              }
              .width('100%')
              TextInput({ text: `${this.settings.diffusionSeed}`, placeholder: '42，-1 表示每次随机' })
                .type(InputType.Number)
                .onChange((value: string) => {
                  const parsed = Number(value);
                  if (value === '' || value === '-' || Number.isNaN(parsed)) {
                    return;
                  }
                  const next = this.copySettings();
                  next.diffusionSeed = Math.trunc(parsed);
                  this.settings = next;
                })
            }
            .width('100%')
            .padding(14)
            .backgroundColor(Color.White)
            .borderRadius(12)

            Column({ space: 8 }) {
              Row() {
                Text('内存模式')
                  .fontSize(15)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(this.memoryModeLabel(this.settings.diffusionMemoryMode))
                  .fontSize(13)
                  .fontColor('#6B7280')
              }
              .width('100%')
              Select([
                { value: '省内存' },
                { value: '均衡' },
                { value: '高性能' }
              ])
                .selected(this.memoryModeIndex(this.settings.diffusionMemoryMode))
                .onSelect((index: number) => {
                  const next = this.copySettings();
                  next.diffusionMemoryMode = index === 0 ? 0 : (index === 1 ? 2 : 1);
                  this.settings = next;
                })
            }
            .width('100%')
            .padding(14)
            .backgroundColor(Color.White)
            .borderRadius(12)
          }

          if (!this.isDiffusion) {
          Column({ space: 8 }) {
            this.settingTitle('后端', this.settings.backendType === RuntimeBackendType.OPENCL ? 'OpenCL' : 'CPU')
            Select([
              { value: 'CPU' },
              { value: 'OpenCL' }
            ])
              .selected(this.settings.backendType === RuntimeBackendType.OPENCL ? 1 : 0)
              .onSelect((index: number) => {
                const next = this.copySettings();
                next.backendType = index === 1 ? RuntimeBackendType.OPENCL : RuntimeBackendType.CPU;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('线程数', this.settings.threadNum.toString())
            Slider({ value: this.settings.threadNum, min: 1, max: 16, step: 1 })
              .onChange((value: number) => {
                const next = this.copySettings();
                next.threadNum = Math.round(value);
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('精度', this.settings.precision)
            Select([
              { value: '低精度' },
              { value: '标准精度' },
              { value: '高精度' }
            ])
              .selected(this.settings.precision === RuntimePrecision.LOW ? 0 :
                (this.settings.precision === RuntimePrecision.NORMAL ? 1 : 2))
              .onSelect((index: number) => {
                const next = this.copySettings();
                next.precision = index === 0 ? RuntimePrecision.LOW :
                  (index === 1 ? RuntimePrecision.NORMAL : RuntimePrecision.HIGH);
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('内存模式', this.settings.memory)
            Select([
              { value: '低内存' },
              { value: '标准' },
              { value: '高性能' }
            ])
              .selected(this.settings.memory === RuntimeMemory.LOW ? 0 :
                (this.settings.memory === RuntimeMemory.NORMAL ? 1 : 2))
              .onSelect((index: number) => {
                const next = this.copySettings();
                next.memory = index === 0 ? RuntimeMemory.LOW :
                  (index === 1 ? RuntimeMemory.NORMAL : RuntimeMemory.HIGH);
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Row() {
            Text('内存映射')
              .fontSize(15)
              .layoutWeight(1)
            Toggle({ type: ToggleType.Switch, isOn: this.settings.useMmap })
              .onChange((value: boolean) => {
                const next = this.copySettings();
                next.useMmap = value;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Row() {
            Text('Prompt Cache')
              .fontSize(15)
              .layoutWeight(1)
            Toggle({ type: ToggleType.Switch, isOn: this.settings.promptCache })
              .onChange((value: boolean) => {
                const next = this.copySettings();
                next.promptCache = value;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Row() {
            Text('思考模式')
              .fontSize(15)
              .layoutWeight(1)
            Toggle({ type: ToggleType.Switch, isOn: this.settings.thinking })
              .onChange((value: boolean) => {
                const next = this.copySettings();
                next.thinking = value;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('最大生成 Token', this.settings.maxTokens.toString())
            Slider({ value: this.settings.maxTokens, min: 64, max: 2048, step: 64 })
              .onChange((value: number) => {
                const next = this.copySettings();
                next.maxTokens = Math.round(value);
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('采样器', this.settings.samplerType)
            Select([
              { value: '混合采样' },
              { value: '贪心' },
              { value: '惩罚采样' },
              { value: '温度' },
              { value: 'Top-K' },
              { value: 'Top-P' }
            ])
              .selected(this.samplerIndex())
              .onSelect((index: number) => {
                const next = this.copySettings();
                const values: RuntimeSamplerType[] = [
                  RuntimeSamplerType.MIXED,
                  RuntimeSamplerType.GREEDY,
                  RuntimeSamplerType.PENALTY,
                  RuntimeSamplerType.TEMPERATURE,
                  RuntimeSamplerType.TOP_K,
                  RuntimeSamplerType.TOP_P
                ];
                next.samplerType = values[index];
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            this.settingTitle('Temperature', this.settings.temperature.toFixed(2))
            Slider({ value: this.settings.temperature, min: 0.05, max: 2, step: 0.05 })
              .enabled(this.settings.samplerType !== RuntimeSamplerType.GREEDY &&
                this.settings.samplerType !== RuntimeSamplerType.TOP_K &&
                this.settings.samplerType !== RuntimeSamplerType.PENALTY)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.temperature = value;
                this.settings = next;
              })
            this.settingTitle('Top-P', this.settings.topP.toFixed(2))
            Slider({ value: this.settings.topP, min: 0, max: 1, step: 0.05 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.MIXED ||
                this.settings.samplerType === RuntimeSamplerType.TOP_P)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.topP = value;
                this.settings = next;
              })
            this.settingTitle('Top-K', this.settings.topK.toString())
            Slider({ value: this.settings.topK, min: 1, max: 200, step: 1 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.MIXED ||
                this.settings.samplerType === RuntimeSamplerType.TOP_K)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.topK = Math.round(value);
                this.settings = next;
              })
            this.settingTitle('Min-P', this.settings.minP.toFixed(2))
            Slider({ value: this.settings.minP, min: 0, max: 1, step: 0.01 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.MIXED)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.minP = value;
                this.settings = next;
              })
            this.settingTitle('TFS-Z', this.settings.tfsZ.toFixed(2))
            Slider({ value: this.settings.tfsZ, min: 0, max: 1, step: 0.01 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.MIXED)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.tfsZ = value;
                this.settings = next;
              })
            this.settingTitle('Typical', this.settings.typical.toFixed(2))
            Slider({ value: this.settings.typical, min: 0, max: 1, step: 0.01 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.MIXED)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.typical = value;
                this.settings = next;
              })
            this.settingTitle('重复惩罚', this.settings.penalty.toFixed(2))
            Slider({ value: this.settings.penalty, min: 1, max: 2, step: 0.01 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.PENALTY)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.penalty = value;
                this.settings = next;
              })
            this.settingTitle('N-Gram', this.settings.nGram.toString())
            Slider({ value: this.settings.nGram, min: 0, max: 16, step: 1 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.PENALTY)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.nGram = Math.round(value);
                this.settings = next;
              })
            this.settingTitle('N-Gram Factor', this.settings.nGramFactor.toFixed(2))
            Slider({ value: this.settings.nGramFactor, min: 0, max: 2, step: 0.01 })
              .enabled(this.settings.samplerType === RuntimeSamplerType.PENALTY)
              .onChange((value: number) => {
                const next = this.copySettings();
                next.nGramFactor = value;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 8 }) {
            Text('系统提示词')
              .fontSize(15)
              .fontWeight(FontWeight.Medium)
              .width('100%')
            TextArea({ text: this.settings.systemPrompt })
              .height(110)
              .onChange((value: string) => {
                const next = this.copySettings();
                next.systemPrompt = value;
                this.settings = next;
              })
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)
          }

          Column({ space: 8 }) {
            Text('模型市场令牌')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor('#171A1F')
              .width('100%')
            Text('公开模型无需令牌；私有仓库需填写平台 access token，仅保存在本机。')
              .fontSize(11)
              .lineHeight(17)
              .fontColor('#7A7F87')
              .width('100%')
            TextInput({ text: this.tokens.huggingFace, placeholder: 'HuggingFace token（hf_…）' })
              .type(InputType.Password)
              .onChange((value: string) => {
                const next = this.copyTokens();
                next.huggingFace = value;
                this.tokens = next;
              })
            TextInput({ text: this.tokens.modelScope, placeholder: 'ModelScope access token' })
              .type(InputType.Password)
              .onChange((value: string) => {
                const next = this.copyTokens();
                next.modelScope = value;
                this.tokens = next;
              })
            TextInput({ text: this.tokens.modelers, placeholder: 'Modelers access token' })
              .type(InputType.Password)
              .onChange((value: string) => {
                const next = this.copyTokens();
                next.modelers = value;
                this.tokens = next;
              })
            Button('保存令牌')
              .height(34)
              .fontSize(13)
              .backgroundColor('#EEF3FF')
              .fontColor('#2864DC')
              .onClick(() => this.saveTokens())
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Column({ space: 10 }) {
            Text('应用设置')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor('#171A1F')
              .width('100%')
            Row() {
              Text('显示性能指标')
                .fontSize(15)
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.showMetrics })
                .onChange((value: boolean) => this.toggleShowMetrics(value))
            }
            .width('100%')
            Row() {
              Text('聊天时暂停下载')
                .fontSize(15)
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.stopDownloadsOnChat })
                .onChange((value: boolean) => this.toggleStopDownloadsOnChat(value))
            }
            .width('100%')
            Row() {
              Text('模型回复语音（需模型支持）')
                .fontSize(15)
                .layoutWeight(1)
              Toggle({ type: ToggleType.Switch, isOn: this.audioReply })
                .onChange((value: boolean) => this.toggleAudioReply(value))
            }
            .width('100%')
            Text('仅对带 talker 的 Omni 模型生效；体积较大的模型合成语音时可能超出系统内存限制')
              .fontSize(11)
              .lineHeight(17)
              .fontColor('#7A7F87')
              .width('100%')
            Row() {
              Text('视频采样帧数')
                .fontSize(15)
                .layoutWeight(1)
              Select([
                { value: '2' },
                { value: '4' },
                { value: '6' },
                { value: '8' }
              ])
                .selected(this.videoFrameIndex(this.videoFrames))
                .onSelect((index: number) => this.updateVideoFrames(this.videoFrameAt(index)))
            }
            .width('100%')
            Row() {
              Text('视频帧尺寸')
                .fontSize(15)
                .layoutWeight(1)
              Select([
                { value: '256' },
                { value: '320' },
                { value: '384' }
              ])
                .selected(this.videoSizeIndex(this.videoFrameSize))
                .onSelect((index: number) => this.updateVideoFrameSize(this.videoSizeAt(index)))
            }
            .width('100%')
            Text('帧数和尺寸越大理解越准，但视频首字延迟越高；默认 4 帧 / 320px')
              .fontSize(11)
              .lineHeight(17)
              .fontColor('#7A7F87')
              .width('100%')
            Text(`缓存占用 ${this.formatBytes(this.cacheBytes)}`)
              .fontSize(14)
              .fontColor('#4B5563')
              .width('100%')
            Button('清理缓存')
              .height(34)
              .fontSize(13)
              .backgroundColor('#F1F3F5')
              .fontColor('#5F6670')
              .onClick(() => this.clearCache())
            Text(this.appVersion ? `版本 ${this.appVersion}` : '版本未知')
              .fontSize(14)
              .fontColor('#4B5563')
              .width('100%')
            Text(this.engineVersion ? `MNN 引擎 ${this.engineVersion}` : 'MNN 引擎版本未知')
              .fontSize(13)
              .fontColor('#6B7280')
              .width('100%')
            Row({ space: 10 }) {
              Button('检查更新')
                .layoutWeight(1)
                .height(34)
                .fontSize(13)
                .backgroundColor('#EEF3FF')
                .fontColor('#2864DC')
                .onClick(() => this.checkForUpdate())
              Button('关于')
                .layoutWeight(1)
                .height(34)
                .fontSize(13)
                .backgroundColor('#F1F3F5')
                .fontColor('#5F6670')
                .onClick(() => this.showAbout())
            }
            .width('100%')
            Text('聊天记录、模型与配置仅保存在本机；API 服务只在启动后监听本机或局域网。')
              .fontSize(12)
              .lineHeight(19)
              .fontColor('#6B7280')
              .width('100%')
          }
          .width('100%')
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)

          Row({ space: 10 }) {
            Button('恢复默认')
              .layoutWeight(1)
              .enabled(!this.busy)
              .onClick(() => this.reset())
            Button('保存设置')
              .layoutWeight(1)
              .enabled(!this.busy)
              .fontColor(Color.White)
              .backgroundColor('#2864DC')
              .onClick(() => this.save())
          }

          if (this.status) {
            Text(this.status)
              .fontSize(13)
              .fontColor('#168653')
              .width('100%')
          }
          if (this.error) {
            Text(`错误：${this.error}`)
              .fontSize(13)
              .fontColor('#C62828')
              .width('100%')
          }
        }
        .padding({ left: 16, right: 16, bottom: 20 })
      }
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}
