import { tr } from '../../services/I18n';
import { common } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import { DownloadRepository } from '../../data/DownloadRepository';
import { ModelRepository } from '../../data/ModelRepository';
import {
  DownloadState,
  DownloadTaskRecord,
  ModelCapabilities,
  ModelRecord,
  ModelState,
  ModelStorageStats
} from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../../services/SettingsStore';
import { DownloadManager } from '../../services/DownloadManager';

class DownloadSummary {
  modelId: string = '';
  downloadedBytes: number = 0;
  totalBytes: number = 0;
  state: string = '';
}

@Component
export struct ModelsPage {
  @Link activeModelId: string;
  onOpenChat: (modelId: string) => void = () => {};
  onOpenSettings: () => void = () => {};
  onStar: () => void = () => {};
  onReportIssue: () => void = () => {};
  /** Bumped by Index when the models tab becomes visible again. */
  @Watch('onRefreshTick') @Prop refreshTick: number = 0;
  @State private fabOpen: boolean = false;
  private settledListener: () => void = () => this.refresh(true);

  aboutToDisappear(): void {
    DownloadManager.getInstance().removeOnSettledListener(this.settledListener);
  }

  private onRefreshTick(): void {
    if (this.refreshTick > 0) {
      this.refresh(true);
    }
  }
  @State private models: ModelRecord[] = [];
  @State private storage: ModelStorageStats = new ModelStorageStats();
  @State private query: string = '';
  @State private downloads: DownloadTaskRecord[] = [];
  @State private caps: Map<string, ModelCapabilities> = new Map<string, ModelCapabilities>();
  @State private filterKind: string = 'all';
  @State private error: string = '';
  @State private busy: boolean = false;
  @State private menuModelId: string = '';

  aboutToAppear(): void {
    this.refresh(true);
    DownloadManager.getInstance().addOnSettledListener(this.settledListener);
  }

  private refresh(scan: boolean = false): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.busy = true;
    AppInitializer.initialize(context).then(() => {
      return scan ? ModelRepository.getInstance().scanModels(context.filesDir) :
        ModelRepository.getInstance().listModels();
    }).then((models: ModelRecord[]) => {
      this.models = models;
      this.storage = ModelRepository.getInstance().getStorageStats(context.filesDir, models);
      return Promise.all(models.map((model: ModelRecord) => ModelRepository.getInstance().getCapabilities(model.id)))
        .then((caps: ModelCapabilities[]) => {
          const map = new Map<string, ModelCapabilities>();
          for (let index = 0; index < models.length; index++) {
            map.set(models[index].id, caps[index]);
          }
          this.caps = map;
          return DownloadRepository.getInstance().list();
        });
    }).then((records: DownloadTaskRecord[]) => {
      this.downloads = records.filter((record: DownloadTaskRecord) => record.state !== DownloadState.COMPLETED);
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private visibleModels(): ModelRecord[] {
    const query = this.query.trim().toLowerCase();
    const matched = query ?
      this.models.filter((model: ModelRecord) =>
        model.name.toLowerCase().indexOf(query) >= 0 || model.configPath.toLowerCase().indexOf(query) >= 0) :
      this.models;
    if (this.filterKind === 'all') {
      return matched;
    }
    return matched.filter((model: ModelRecord) => {
      const caps = this.caps.get(model.id);
      if (caps === undefined) {
        return false;
      }
      return this.filterKind === 'vision' ? caps.vision : caps.audioInput;
    });
  }

  private async importModel(): Promise<void> {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const options = new picker.DocumentSelectOptions();
    options.maxSelectNumber = 1;
    options.selectMode = picker.DocumentSelectMode.FOLDER;
    try {
      const uris = await new picker.DocumentViewPicker(context).select(options);
      if (uris.length === 0) {
        return;
      }
      this.busy = true;
      await ModelRepository.getInstance().importModel(uris[0], context.filesDir, context.cacheDir);
      this.refresh(true);
    } catch (error) {
      this.error = (error as Error).message;
      this.busy = false;
    }
  }

  private useModel(model: ModelRecord): void {
    if (model.state !== ModelState.READY || this.busy) {
      return;
    }
    SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, model.id).then(() => {
      this.activeModelId = model.id;
      this.onOpenChat(model.id);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private menuModel(): ModelRecord | undefined {
    return this.models.find((model: ModelRecord) => model.id === this.menuModelId);
  }

  // No-parameter builder: bindContextMenu does not fire with a closure builder inside ForEach,
  // so the long-pressed model is recorded via onTouch instead (same pattern as MessageBubble).
  @Builder
  private modelMenu() {
    Menu() {
      MenuItem({ content: tr('打开聊天') })
        .onClick(() => {
          const model = this.menuModel();
          if (model) {
            this.useModel(model);
          }
        })
      MenuItem({ content: tr('模型设置') })
        .onClick(() => this.onOpenSettings())
      MenuItem({ content: tr('删除') })
        .onClick(() => {
          const model = this.menuModel();
          if (model) {
            this.confirmDelete(model);
          }
        })
    }
  }

  private togglePinned(model: ModelRecord): void {
    ModelRepository.getInstance().setPinned(model.id, !model.isPinned).then(() => {
      this.refresh();
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private confirmDelete(model: ModelRecord): void {
    AlertDialog.show({
      title: tr('删除模型'),
      message: `${tr('确认删除 ')}${model.name}${tr('？此操作会移除受管模型文件。')}`,
      primaryButton: {
        value: tr('取消'),
        action: () => {
        }
      },
      secondaryButton: {
        value: tr('删除'),
        fontColor: $r('app.color.danger'),
        action: () => this.deleteModel(model)
      }
    });
  }

  private deleteModel(model: ModelRecord): void {
    if (model.isProtected || model.isLegacy || this.activeModelId === model.id) {
      return;
    }
    this.busy = true;
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    ModelRepository.getInstance().deleteModel(model.id, context.filesDir)
      .then(() => SettingsStore.getInstance().resetModelRuntimeSettings(model.id))
      .then(() => {
        this.refresh(true);
      }).catch((error: Error) => {
      this.error = error.message;
      this.busy = false;
    });
  }

  private downloadSummaries(): DownloadSummary[] {
    const summaries: DownloadSummary[] = [];
    for (let index = 0; index < this.downloads.length; index++) {
      const record = this.downloads[index];
      let summary = summaries.find((item: DownloadSummary) => item.modelId === record.modelId);
      if (summary === undefined) {
        summary = new DownloadSummary();
        summary.modelId = record.modelId;
        summaries.push(summary);
      }
      summary.downloadedBytes += record.downloadedBytes;
      summary.totalBytes += record.totalBytes;
      if (record.state === DownloadState.PAUSED || record.state === DownloadState.FAILED) {
        summary.state = record.state;
      } else if (summary.state !== DownloadState.PAUSED && summary.state !== DownloadState.FAILED) {
        summary.state = record.state;
      }
    }
    return summaries;
  }

  private cancelDownload(modelId: string): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.busy = true;
    DownloadManager.getInstance().cancel(context, modelId).then(() => {
      this.refresh(false);
    }).catch((error: Error) => {
      this.error = error.message;
      this.busy = false;
    });
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  private chips(model: ModelRecord): string[] {
    const caps = this.caps.get(model.id);
    const chips: string[] = [];
    if (caps) {
      chips.push(caps.diffusion ? 'image' : 'text');
      if (caps.vision) {
        chips.push('vision');
      }
      if (caps.audioInput || caps.audioOutput) {
        chips.push('audio');
      }
    }
    return chips;
  }

  private statusLine(model: ModelRecord): string {
    if (model.state !== ModelState.READY) {
      return tr('配置无效');
    }
    const suffix = this.activeModelId === model.id ? `, ${tr('使用中')}` : '';
    return `${this.formatBytes(model.sizeBytes)}, ${tr('已下载')}${suffix}`;
  }

  @Builder
  private fabItem(icon: string, label: string, action: () => void) {
    Row({ space: 10 }) {
      Text(label)
        .fontSize(13)
        .fontColor($r('app.color.text_primary'))
        .padding({ left: 12, right: 12, top: 7, bottom: 7 })
        .borderRadius(16)
        .backgroundColor($r('app.color.surface'))
        .shadow({ radius: 8, color: '#22000000', offsetY: 2 })
      Button() {
        Text(icon)
          .fontSize(18)
          .fontColor(Color.White)
      }
      .width(48)
      .height(48)
      .borderRadius(16)
      .backgroundColor($r('app.color.brand'))
      .shadow({ radius: 8, color: '#22000000', offsetY: 2 })
      .onClick(action)
    }
  }

  build() {
    Stack({ alignContent: Alignment.BottomEnd }) {
      Column() {
        TextInput({ text: this.query, placeholder: tr('搜索名称或路径') })
          .backgroundColor($r('app.color.surface'))
          .borderRadius(22)
          .margin({ left: 16, right: 16, top: 10, bottom: 6 })
          .onChange((value: string) => {
            this.query = value;
          })

        Row({ space: 8 }) {
          Button(tr('全部'))
            .height(30)
            .fontSize(12)
            .backgroundColor(this.filterKind === 'all' ? $r('app.color.brand_tint') : $r('app.color.surface'))
            .fontColor(this.filterKind === 'all' ? $r('app.color.chip_text') : $r('app.color.text_muted'))
            .onClick(() => this.filterKind = 'all')
          Button(tr('视觉'))
            .height(30)
            .fontSize(12)
            .backgroundColor(this.filterKind === 'vision' ? $r('app.color.brand_tint') : $r('app.color.surface'))
            .fontColor(this.filterKind === 'vision' ? $r('app.color.chip_text') : $r('app.color.text_muted'))
            .onClick(() => this.filterKind = 'vision')
          Button(tr('音频'))
            .height(30)
            .fontSize(12)
            .backgroundColor(this.filterKind === 'audio' ? $r('app.color.brand_tint') : $r('app.color.surface'))
            .fontColor(this.filterKind === 'audio' ? $r('app.color.chip_text') : $r('app.color.text_muted'))
            .onClick(() => this.filterKind = 'audio')
          Blank()
          Button(tr('扫描'))
            .height(30)
            .fontSize(12)
            .backgroundColor(Color.Transparent)
            .fontColor($r('app.color.text_secondary'))
            .enabled(!this.busy)
            .onClick(() => this.refresh(true))
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 6 })

      if (this.downloads.length > 0) {
        Column({ space: 6 }) {
          Text(tr('下载中'))
            .fontSize(13)
            .fontColor($r('app.color.text_secondary'))
            .width('100%')
          ForEach(this.downloadSummaries(), (item: DownloadSummary) => {
            Row({ space: 10 }) {
              Column({ space: 4 }) {
                Text(item.modelId)
                  .fontSize(14)
                  .fontColor($r('app.color.text_primary'))
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                Progress({
                  value: item.totalBytes > 0 ? item.downloadedBytes : 0,
                  total: item.totalBytes > 0 ? item.totalBytes : 1,
                  type: ProgressType.Linear
                })
                  .width('100%')
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
              Text(item.totalBytes > 0 ? `${((item.downloadedBytes / item.totalBytes) * 100).toFixed(1)}%` : tr('准备中'))
                .fontSize(12)
                .fontColor($r('app.color.text_secondary'))
              Button(tr('取消'))
                .height(30)
                .fontSize(12)
                .enabled(!this.busy)
                .backgroundColor($r('app.color.danger_tint'))
                .fontColor($r('app.color.danger'))
                .onClick(() => this.cancelDownload(item.modelId))
            }
            .width('100%')
            .padding(12)
            .borderRadius(12)
            .backgroundColor($r('app.color.surface'))
          }, (item: DownloadSummary) => item.modelId)
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 10 })
      }

      List() {
        ForEach(this.visibleModels(), (model: ModelRecord) => {
          ListItem() {
            // Android-style model row: logo tile, name, tag chips, "size, Downloaded" status.
            Row({ space: 14 }) {
              Image($r('app.media.icon'))
                .width(56)
                .height(56)
                .borderRadius(14)
                .objectFit(ImageFit.Cover)
              Column({ space: 6 }) {
                Text(model.name)
                  .fontSize(17)
                  .fontWeight(FontWeight.Medium)
                  .fontColor($r('app.color.text_primary'))
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                  .width('100%')
                Row({ space: 6 }) {
                  ForEach(this.chips(model), (chip: string) => {
                    Text(chip)
                      .fontSize(12)
                      .fontColor($r('app.color.chip_text'))
                      .backgroundColor($r('app.color.chip_bg'))
                      .padding({ left: 10, right: 10, top: 4, bottom: 4 })
                      .borderRadius(12)
                  }, (chip: string) => chip)
                }
                Text(this.statusLine(model))
                  .fontSize(13)
                  .fontColor(model.state === ModelState.READY ?
                    $r('app.color.text_secondary') : $r('app.color.danger'))
              }
              .layoutWeight(1)
              .alignItems(HorizontalAlign.Start)

              if (model.isPinned) {
                Text('★')
                  .fontSize(16)
                  .fontColor($r('app.color.brand'))
              }
            }
            .width('100%')
            .padding({ left: 16, right: 16, top: 12, bottom: 12 })
            .onClick(() => this.useModel(model))
            .onTouch((event: TouchEvent) => {
              if (event.type === TouchType.Down) {
                this.menuModelId = model.id;
              }
            })
            .bindContextMenu(this.modelMenu, ResponseType.LongPress)
          }
        }, (model: ModelRecord) => model.id)
      }
      .divider({ strokeWidth: 1, color: $r('app.color.divider_soft'), startMargin: 86, endMargin: 16 })
      .layoutWeight(1)

      if (this.error) {
        Text(`${tr('错误：')}${this.error}`)
          .fontSize(13)
          .fontColor($r('app.color.danger'))
          .padding(16)
      }
      }
      .width('100%')
      .height('100%')

      // Android-style ExpandableFab: the "+" opens sub-actions for Star / issue / import.
      Column({ space: 14 }) {
        if (this.fabOpen) {
          this.fabItem('★', tr('给 MNN 点个 Star'), () => {
            this.fabOpen = false;
            this.onStar();
          })
          this.fabItem('⚑', tr('上报问题'), () => {
            this.fabOpen = false;
            this.onReportIssue();
          })
          this.fabItem('⤓', tr('导入本地模型'), () => {
            this.fabOpen = false;
            this.importModel();
          })
        }
        Button() {
          Text(this.fabOpen ? '×' : '+')
            .fontSize(28)
            .fontColor(Color.White)
        }
        .width(64)
        .height(64)
        .borderRadius(20)
        .backgroundColor($r('app.color.brand'))
        .shadow({ radius: 12, color: '#33000000', offsetY: 4 })
        .enabled(!this.busy)
        .onClick(() => {
          this.fabOpen = !this.fabOpen;
        })
      }
      .alignItems(HorizontalAlign.End)
      .margin({ right: 20, bottom: 24 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
  }
}
