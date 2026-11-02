import { common } from '@kit.AbilityKit';
import { ModelRepository } from '../../data/ModelRepository';
import { ModelState, RuntimeBackendType } from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { BenchmarkService, BenchmarkSummary } from '../../services/BenchmarkService';
import { GenerationCoordinator, NativeBenchmarkSample } from '../../services/GenerationCoordinator';

@Component
export struct BenchmarkPage {
  @Watch('onConfigurationChanged') @Link activeModelId: string;
  @Watch('onConfigurationChanged') @Link settingsRevision: number;
  @State private modelName: string = '';
  @State private backendType: RuntimeBackendType = RuntimeBackendType.OPENCL;
  @State private ready: boolean = false;
  @State private running: boolean = false;
  @State private iteration: number = 0;
  @State private prefillSpeed: number = 0;
  @State private decodeSpeed: number = 0;
  @State private summary: BenchmarkSummary = new BenchmarkSummary();
  @State private error: string = '';
  private requestId: string = '';

  aboutToAppear(): void {
    this.refreshModel();
  }

  private onConfigurationChanged(): void {
    if (!this.running) {
      this.summary = new BenchmarkSummary();
      this.refreshModel();
    }
  }

  private refreshModel(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(() => {
      return ModelRepository.getInstance().getModel(this.activeModelId);
    }).then((model) => {
      this.modelName = model?.name || this.activeModelId;
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
    this.requestId = GenerationCoordinator.getInstance().createRequestId('benchmark');
    this.running = true;
    this.iteration = 0;
    this.error = '';
    this.summary = new BenchmarkSummary();
    BenchmarkService.getInstance().run(
      context, this.activeModelId, this.settingsRevision, this.requestId, this.backendType,
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
        Text(this.modelName || this.activeModelId)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .width('100%')
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
        Text(`${this.backendType === RuntimeBackendType.OPENCL ? 'OpenCL' : 'CPU'} · ` +
          'PP 128 · TG 128 · 预热 1 次 · 测量 3 次')
          .fontSize(13)
          .fontColor('#6B7280')
          .width('100%')
        if (this.running) {
          Progress({ value: this.iteration, total: 3, type: ProgressType.Linear })
            .width('100%')
          Text(this.iteration === 0 ? '正在预热' : `正在运行 ${this.iteration}/3`)
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
