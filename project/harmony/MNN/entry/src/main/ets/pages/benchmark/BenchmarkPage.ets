import { common } from '@kit.AbilityKit';
import pasteboard from '@ohos.pasteboard';
import { ModelPickerSheet } from '../../components/ModelPickerSheet';
import { ModelRepository } from '../../data/ModelRepository';
import { ModelRecord, ModelState, RuntimeBackendType } from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { BenchmarkService, BenchmarkSummary } from '../../services/BenchmarkService';
import { GenerationCoordinator, NativeBenchmarkSample } from '../../services/GenerationCoordinator';

@Component
export struct BenchmarkPage {
  @Watch('onConfigurationChanged') @Link activeModelId: string;
  @Watch('onConfigurationChanged') @Link settingsRevision: number;
  @State private modelName: string = '';
  @State private benchModelId: string = '';
  @State private promptTokensText: string = '128';
  @State private generateTokensText: string = '128';
  @State private repeatsText: string = '3';
  @State private backendType: RuntimeBackendType = RuntimeBackendType.CPU;
  @State private ready: boolean = false;
  @State private running: boolean = false;
  @State private iteration: number = 0;
  @State private prefillSpeed: number = 0;
  @State private decodeSpeed: number = 0;
  @State private summary: BenchmarkSummary = new BenchmarkSummary();
  @State private error: string = '';
  private requestId: string = '';

  aboutToAppear(): void {
    this.benchModelId = this.activeModelId;
    this.refreshModel();
  }

  private onConfigurationChanged(): void {
    if (!this.running) {
      this.benchModelId = this.activeModelId;
      this.summary = new BenchmarkSummary();
      this.refreshModel();
    }
  }

  private openModelPicker(): void {
    if (this.running) {
      return;
    }
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const ready: ModelRecord[] = models.filter((model: ModelRecord) => model.state === ModelState.READY);
      if (ready.length === 0) {
        this.error = '没有可用的模型';
        return;
      }
      let controller: CustomDialogController | null = null;
      controller = new CustomDialogController({
        builder: ModelPickerSheet({
          models: ready,
          activeModelId: this.benchModelId,
          onPick: (model: ModelRecord) => {
            this.benchModelId = model.id;
            this.modelName = model.name;
            this.ready = true;
            this.summary = new BenchmarkSummary();
          }
        }),
        autoCancel: true,
        alignment: DialogAlignment.Bottom,
        customStyle: true
      });
      controller.open();
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private clampInt(text: string, fallback: number, minimum: number, maximum: number): number {
    const value = Number(text);
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return Math.min(maximum, Math.max(minimum, Math.round(value)));
  }

  private shareSummary(): void {
    const text = `MNN Benchmark (${this.summary.modelId})\n` +
      `backend: ${this.summary.backendType}\n` +
      `PP ${this.summary.promptTokens} / TG ${this.summary.generateTokens} / repeats ${this.summary.repeats}\n` +
      `prefill: ${this.summary.meanPrefillSpeed.toFixed(1)} ± ${this.summary.prefillStdDev.toFixed(1)} tok/s\n` +
      `decode: ${this.summary.meanDecodeSpeed.toFixed(1)} ± ${this.summary.decodeStdDev.toFixed(1)} tok/s\n` +
      `load: ${(this.summary.loadMs / 1000).toFixed(2)}s, total: ${(this.summary.totalMs / 1000).toFixed(2)}s`;
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, text);
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        this.error = '';
      });
    } catch (error) {
      this.error = (error as Error).message;
    }
  }

  private refreshModel(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(() => {
      return ModelRepository.getInstance().getModel(this.activeModelId);
    }).then((model) => {
      this.modelName = model?.name || this.benchModelId;
      this.ready = model?.state === ModelState.READY;
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private run(): void {
    if (!this.ready || this.running) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const promptTokens = this.clampInt(this.promptTokensText, 128, 16, 2048);
    const generateTokens = this.clampInt(this.generateTokensText, 128, 16, 1024);
    const repeats = this.clampInt(this.repeatsText, 3, 1, 10);
    this.requestId = GenerationCoordinator.getInstance().createRequestId('benchmark');
    this.running = true;
    this.iteration = 0;
    this.error = '';
    this.summary = new BenchmarkSummary();
    BenchmarkService.getInstance().run(
      context, this.benchModelId, this.settingsRevision, this.requestId, this.backendType,
      promptTokens, generateTokens, repeats,
      (sample: NativeBenchmarkSample) => {
        this.iteration = sample.iteration;
        this.prefillSpeed = sample.prefillMs > 0 ? sample.promptTokens * 1000 / sample.prefillMs : 0;
        this.decodeSpeed = sample.decodeMs > 0 ? sample.generatedTokens * 1000 / sample.decodeMs : 0;
      }
    ).then((summary: BenchmarkSummary) => {
      this.summary = summary;
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.running = false;
      this.requestId = '';
    });
  }

  private stop(): void {
    if (this.requestId) {
      BenchmarkService.getInstance().cancel(this.requestId);
    }
  }

  build() {
    Column() {
      Text('Benchmark')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .width('100%')
        .padding({ left: 20, right: 20, top: 18, bottom: 12 })

      Column({ space: 10 }) {
        Row({ space: 6 }) {
          Text(this.modelName || this.benchModelId)
            .fontSize(18)
            .fontWeight(FontWeight.Medium)
            .layoutWeight(1)
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
          Text('▾')
            .fontSize(14)
            .fontColor('#6B7280')
        }
        .width('100%')
        .onClick(() => this.openModelPicker())
        Row() {
          Text('后端')
            .fontSize(14)
            .layoutWeight(1)
          Select([
            { value: 'OpenCL' },
            { value: 'CPU' }
          ])
            .selected(this.backendType === RuntimeBackendType.OPENCL ? 0 : 1)
            .enabled(!this.running)
            .onSelect((index: number) => {
              this.backendType = index === 0 ? RuntimeBackendType.OPENCL : RuntimeBackendType.CPU;
              this.summary = new BenchmarkSummary();
            })
        }
        .width('100%')
        Row({ space: 8 }) {
          Text('PP').fontSize(13).fontColor('#6B7280')
          TextInput({ text: this.promptTokensText })
            .type(InputType.Number)
            .enabled(!this.running)
            .layoutWeight(1)
            .onChange((value: string) => this.promptTokensText = value)
          Text('TG').fontSize(13).fontColor('#6B7280')
          TextInput({ text: this.generateTokensText })
            .type(InputType.Number)
            .enabled(!this.running)
            .layoutWeight(1)
            .onChange((value: string) => this.generateTokensText = value)
          Text('轮数').fontSize(13).fontColor('#6B7280')
          TextInput({ text: this.repeatsText })
            .type(InputType.Number)
            .enabled(!this.running)
            .layoutWeight(1)
            .onChange((value: string) => this.repeatsText = value)
        }
        .width('100%')
        if (this.running) {
          Progress({ value: this.iteration, total: this.clampInt(this.repeatsText, 3, 1, 10), type: ProgressType.Linear })
            .width('100%')
          Text(this.iteration === 0 ? '正在预热' :
            `正在运行 ${this.iteration}/${this.clampInt(this.repeatsText, 3, 1, 10)}`)
            .fontSize(14)
            .fontColor('#2864DC')
        }
        Row({ space: 10 }) {
          Button(this.running ? '停止' : (this.summary.samples.length > 0 ? '重新测试' : '开始测试'))
            .layoutWeight(1)
            .enabled(this.ready)
            .fontColor(this.running ? '#C62828' : Color.White)
            .backgroundColor(this.running ? '#FDECEC' : '#2864DC')
            .onClick(() => this.running ? this.stop() : this.run())
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 14 })

      if (this.running || this.iteration > 0) {
        Row({ space: 10 }) {
          Column({ space: 5 }) {
            Text('Prefill')
              .fontSize(12)
              .fontColor('#6B7280')
            Text(`${this.prefillSpeed.toFixed(1)} tok/s`)
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
          }
          .layoutWeight(1)
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)
          Column({ space: 5 }) {
            Text('Decode')
              .fontSize(12)
              .fontColor('#6B7280')
            Text(`${this.decodeSpeed.toFixed(1)} tok/s`)
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
          }
          .layoutWeight(1)
          .padding(14)
          .backgroundColor(Color.White)
          .borderRadius(12)
        }
        .padding({ left: 16, right: 16, bottom: 14 })
      }

      if (this.summary.samples.length > 0) {
        Column({ space: 9 }) {
          Text(`${this.summary.backendType === RuntimeBackendType.OPENCL ? 'OpenCL' : 'CPU'} · ` +
            (this.summary.canceled ? '测试已停止' : '测试完成'))
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .width('100%')
          Text(`Prefill ${this.summary.meanPrefillSpeed.toFixed(1)} ± ` +
            `${this.summary.prefillStdDev.toFixed(1)} tok/s`)
            .fontSize(15)
            .width('100%')
          Text(`Decode ${this.summary.meanDecodeSpeed.toFixed(1)} ± ` +
            `${this.summary.decodeStdDev.toFixed(1)} tok/s`)
            .fontSize(15)
            .width('100%')
          Text(`测量总耗时 ${(this.summary.totalMs / 1000).toFixed(2)}s · ` +
            `模型加载 ${(this.summary.loadMs / 1000).toFixed(2)}s`)
            .fontSize(13)
            .fontColor('#6B7280')
            .width('100%')
          ForEach(this.summary.samples, (sample: NativeBenchmarkSample) => {
            Text(`第 ${sample.iteration} 轮：prefill ${(sample.prefillMs > 0 ? sample.promptTokens * 1000 / sample.prefillMs : 0).toFixed(1)} tok/s · ` +
              `decode ${(sample.decodeMs > 0 ? sample.generatedTokens * 1000 / sample.decodeMs : 0).toFixed(1)} tok/s`)
              .fontSize(12)
              .fontColor('#6B7280')
              .width('100%')
          }, (sample: NativeBenchmarkSample) => sample.iteration.toString())
          Button('复制结果')
            .height(34)
            .fontSize(13)
            .backgroundColor('#EEF3FF')
            .fontColor('#2864DC')
            .onClick(() => this.shareSummary())
        }
        .padding(16)
        .backgroundColor(Color.White)
        .borderRadius(14)
        .margin({ left: 16, right: 16 })
      }

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .padding(16)
      }

      Blank()
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}
