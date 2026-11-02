import { ModelRepository } from '../../data/ModelRepository';
import {
  ModelRuntimeSettings,
  RuntimeBackendType,
  RuntimeMemory,
  RuntimePrecision,
  RuntimeSamplerType
} from '../../domain/AppTypes';
import { SettingsStore } from '../../services/SettingsStore';

@Component
export struct SettingsPage {
  @Watch('onActiveModelChanged') @Link activeModelId: string;
  @Link settingsRevision: number;
  @State private modelName: string = '';
  @State private settings: ModelRuntimeSettings = new ModelRuntimeSettings();
  @State private status: string = '';
  @State private error: string = '';
  @State private busy: boolean = false;
  private loadSequence: number = 0;

  aboutToAppear(): void {
    this.loadSettings();
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
    ModelRepository.getInstance().getModel(modelId).then((model) => {
      const modelName = model?.name || modelId;
      return SettingsStore.getInstance().getModelRuntimeSettings(modelId).then((settings: ModelRuntimeSettings) => {
        if (sequence !== this.loadSequence || modelId !== this.activeModelId) {
          return;
        }
        this.modelName = modelName;
        this.settings = settings;
        this.status = '';
        this.error = '';
      });
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
    return copy;
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
              { value: '温度' },
              { value: 'Top-K' },
              { value: 'Top-P' }
            ])
              .selected(this.settings.samplerType === RuntimeSamplerType.MIXED ? 0 :
                (this.settings.samplerType === RuntimeSamplerType.GREEDY ? 1 :
                  (this.settings.samplerType === RuntimeSamplerType.TEMPERATURE ? 2 :
                    (this.settings.samplerType === RuntimeSamplerType.TOP_K ? 3 : 4))))
              .onSelect((index: number) => {
                const next = this.copySettings();
                const values: RuntimeSamplerType[] = [
                  RuntimeSamplerType.MIXED,
                  RuntimeSamplerType.GREEDY,
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
                this.settings.samplerType !== RuntimeSamplerType.TOP_K)
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
