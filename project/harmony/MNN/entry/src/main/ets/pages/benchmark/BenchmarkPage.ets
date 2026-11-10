import { common } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import pasteboard from '@ohos.pasteboard';
import { ModelPickerSheet } from '../../components/ModelPickerSheet';
import { ModelRepository } from '../../data/ModelRepository';
import { ModelRecord, ModelState, RuntimeBackendType } from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { BenchmarkService, BenchmarkSummary } from '../../services/BenchmarkService';
import { GenerationCoordinator, NativeBenchmarkSample } from '../../services/GenerationCoordinator';
import { LeaderboardOutcome, LeaderboardService } from '../../services/LeaderboardService';
import { DatasetItemResult, DatasetTestResult, DatasetTestService } from '../../services/DatasetTestService';

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
  @State private userIdText: string = '';
  @State private submitting: boolean = false;
  @State private leaderboardMessage: string = '';
  @State private leaderboardFailed: boolean = false;
  @State private datasetName: string = '';
  @State private datasetPrompts: string[] = [];
  @State private datasetRunning: boolean = false;
  @State private datasetMessage: string = '';
  @State private datasetResult: DatasetTestResult = new DatasetTestResult();
  private requestId: string = '';

  aboutToAppear(): void {
    this.benchModelId = this.activeModelId;
    this.refreshModel();
    LeaderboardService.getInstance().userId().then((id: string) => {
      this.userIdText = id;
    }).catch(() => {
    });
  }

  private submitScore(): void {
    if (this.submitting || this.summary.samples.length === 0) {
      return;
    }
    this.submitting = true;
    this.leaderboardMessage = '正在提交…';
    this.leaderboardFailed = false;
    const service = LeaderboardService.getInstance();
    service.setUserId(this.userIdText).then(() => {
      return service.submit(this.modelName || this.summary.modelId, this.summary.meanPrefillSpeed,
        this.summary.meanDecodeSpeed, this.summary.peakMemoryKb / 1024);
    }).then((outcome: LeaderboardOutcome) => {
      this.leaderboardFailed = !outcome.success;
      this.leaderboardMessage = outcome.message;
    }).catch((error: Error) => {
      this.leaderboardFailed = true;
      this.leaderboardMessage = error.message;
    }).finally(() => {
      this.submitting = false;
    });
  }

  private chooseDataset(): void {
    if (this.datasetRunning) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const options = new picker.DocumentSelectOptions();
    options.maxSelectNumber = 1;
    options.fileSuffixFilters = ['.jsonl', '.json'];
    new picker.DocumentViewPicker(context).select(options).then((uris: string[]) => {
      if (uris.length === 0) {
        return;
      }
      this.datasetPrompts = DatasetTestService.getInstance().readPrompts(uris[0]);
      const slash = uris[0].lastIndexOf('/');
      this.datasetName = slash >= 0 ? uris[0].substring(slash + 1) : uris[0];
      this.datasetResult = new DatasetTestResult();
      this.datasetMessage = `已加载 ${this.datasetPrompts.length} 条 prompt（最多运行 3 条）`;
    }).catch((error: Error) => {
      this.datasetMessage = error.message;
    });
  }

  private runDataset(): void {
    if (this.datasetRunning || !this.ready || this.datasetPrompts.length === 0) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.datasetRunning = true;
    this.datasetResult = new DatasetTestResult();
    this.datasetMessage = '正在准备模型…';
    DatasetTestService.getInstance().run(context, this.benchModelId, this.datasetName, this.datasetPrompts,
      (index: number, total: number) => {
        this.datasetMessage = `正在测试 ${index}/${total}`;
      }).then((result: DatasetTestResult) => {
      this.datasetResult = result;
      this.datasetMessage = result.canceled ? '测试已停止' : '测试完成';
    }).catch((error: Error) => {
      this.datasetMessage = error.message;
    }).finally(() => {
      this.datasetRunning = false;
    });
  }

  private stopDataset(): void {
    DatasetTestService.getInstance().cancel();
  }

  private copyDatasetResult(): void {
    const lines: string[] = [];
    lines.push(`Dataset: ${this.datasetResult.fileName} (${this.datasetResult.modelId})`);
    this.datasetResult.items.forEach((item: DatasetItemResult) => {
      const prefill = item.prefillMs > 0 ? (item.promptTokens * 1000 / item.prefillMs).toFixed(1) : '0';
      const decode = item.decodeMs > 0 ? (item.generatedTokens * 1000 / item.decodeMs).toFixed(1) : '0';
      lines.push(`#${item.index} prompt ${item.promptTokens} tok, prefill ${prefill} tok/s, ` +
        `decode ${decode} tok/s`);
      lines.push(item.error ? `error: ${item.error}` : item.output);
    });
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, lines.join('\n'));
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        this.datasetMessage = '结果已复制';
      }).catch(() => {
        this.datasetMessage = '复制失败';
      });
    } catch (error) {
      this.datasetMessage = (error as Error).message;
    }
  }

  private queryRank(): void {
    if (this.submitting || this.summary.samples.length === 0) {
      return;
    }
    this.submitting = true;
    this.leaderboardMessage = '正在查询排名…';
    this.leaderboardFailed = false;
    const service = LeaderboardService.getInstance();
    service.setUserId(this.userIdText).then(() => {
      return service.queryRank(this.modelName || this.summary.modelId);
    }).then((outcome: LeaderboardOutcome) => {
      this.leaderboardFailed = !outcome.success;
      this.leaderboardMessage = outcome.message;
    }).catch((error: Error) => {
      this.leaderboardFailed = true;
      this.leaderboardMessage = error.message;
    }).finally(() => {
      this.submitting = false;
    });
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
      `load: ${(this.summary.loadMs / 1000).toFixed(2)}s, total: ${(this.summary.totalMs / 1000).toFixed(2)}s` +
      (this.summary.peakMemoryKb > 0 ? `\npeak memory: ${(this.summary.peakMemoryKb / 1024).toFixed(1)} MB` : '');
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
    Scroll() {
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
            if (this.summary.peakMemoryKb > 0) {
              Text(`峰值内存 ${(this.summary.peakMemoryKb / 1024).toFixed(1)} MB`)
                .fontSize(13)
                .fontColor('#6B7280')
                .width('100%')
            }
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

        Column({ space: 9 }) {
          Text('数据集测试')
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .width('100%')
          Text('选择 JSONL 数据集文件，每行形如 {"prompt": "..."}，最多运行 3 条。')
            .fontSize(11)
            .fontColor('#7A7F87')
            .width('100%')
          Row({ space: 10 }) {
            Button(this.datasetName || '选择数据集文件')
              .layoutWeight(1)
              .height(38)
              .fontSize(13)
              .backgroundColor('#EEF3FF')
              .fontColor('#2864DC')
              .enabled(!this.datasetRunning)
              .onClick(() => this.chooseDataset())
            Button(this.datasetRunning ? '停止' : '开始测试')
              .layoutWeight(1)
              .height(38)
              .fontSize(13)
              .fontColor(this.datasetRunning ? '#C62828' : Color.White)
              .backgroundColor(this.datasetRunning ? '#FDECEC' : '#2864DC')
              .enabled(this.datasetRunning || (this.ready && this.datasetPrompts.length > 0))
              .onClick(() => this.datasetRunning ? this.stopDataset() : this.runDataset())
          }
          .width('100%')
          if (this.datasetMessage) {
            Text(this.datasetMessage)
              .fontSize(12)
              .fontColor('#2864DC')
              .width('100%')
          }
          ForEach(this.datasetResult.items, (item: DatasetItemResult) => {
            Column({ space: 2 }) {
              Text(`#${item.index} · prompt ${item.promptTokens} tok · ` +
                `prefill ${(item.prefillMs > 0 ? item.promptTokens * 1000 / item.prefillMs : 0).toFixed(1)} tok/s · ` +
                `decode ${(item.decodeMs > 0 ? item.generatedTokens * 1000 / item.decodeMs : 0).toFixed(1)} tok/s`)
                .fontSize(12)
                .fontColor('#4B5563')
                .width('100%')
              Text(item.error ? `错误：${item.error}` : item.output)
                .fontSize(12)
                .fontColor(item.error ? '#C62828' : '#171A1F')
                .maxLines(3)
                .textOverflow({ overflow: TextOverflow.Ellipsis })
                .width('100%')
            }
            .width('100%')
          }, (item: DatasetItemResult) => item.index.toString())
          if (this.datasetResult.items.length > 0) {
            Button('复制结果')
              .height(34)
              .fontSize(13)
              .backgroundColor('#EEF3FF')
              .fontColor('#2864DC')
              .onClick(() => this.copyDatasetResult())
          }
        }
        .padding(16)
        .backgroundColor(Color.White)
        .borderRadius(14)
        .margin({ left: 16, right: 16 })

        if (this.summary.samples.length > 0) {
          Column({ space: 9 }) {
            Text('排行榜')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .width('100%')
            Row({ space: 10 }) {
              Text('用户 ID')
                .fontSize(14)
                .fontColor('#4B5563')
              TextInput({ text: this.userIdText, placeholder: '排行榜用户 ID' })
                .layoutWeight(1)
                .height(38)
                .fontSize(13)
                .enabled(!this.submitting)
                .onChange((value: string) => this.userIdText = value)
            }
            .width('100%')
            Text('仅提交预填充/解码速度和峰值内存，不包含聊天内容。')
              .fontSize(11)
              .fontColor('#7A7F87')
              .width('100%')
            Row({ space: 10 }) {
              Button('提交成绩')
                .layoutWeight(1)
                .height(38)
                .fontSize(13)
                .enabled(!this.submitting)
                .onClick(() => this.submitScore())
              Button('查询排名')
                .layoutWeight(1)
                .height(38)
                .fontSize(13)
                .backgroundColor('#EEF3FF')
                .fontColor('#2864DC')
                .enabled(!this.submitting)
                .onClick(() => this.queryRank())
            }
            .width('100%')
            if (this.leaderboardMessage) {
              Text(this.leaderboardMessage)
                .fontSize(12)
                .fontColor(this.leaderboardFailed ? '#C62828' : '#168653')
                .width('100%')
            }
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
      }
      .width('100%')
      .padding({ bottom: 20 })
      .constraintSize({ minHeight: '100%' })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
    .scrollBar(BarState.Off)
  }
}
