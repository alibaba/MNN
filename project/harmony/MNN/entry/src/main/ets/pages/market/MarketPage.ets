import { common } from '@kit.AbilityKit';
import fs from '@ohos.file.fs';
import { DownloadRepository } from '../../data/DownloadRepository';
import {
  DownloadState,
  DownloadTaskRecord,
  MarketCatalogRecord,
  MarketModelRecord,
  ModelDownloadStatus
} from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { DownloadManager } from '../../services/DownloadManager';
import { MarketDetailSheet } from '../../components/MarketDetailSheet';
import { ModelMarketService } from '../../services/ModelMarketService';
import {
  PROVIDER_HUGGING_FACE,
  PROVIDER_MODELERS,
  PROVIDER_MODEL_SCOPE,
  ProviderAdapters
} from '../../services/ProviderAdapters';

@Component
export struct MarketPage {
  @State private catalog: MarketCatalogRecord = new MarketCatalogRecord();
  @State private models: MarketModelRecord[] = [];
  @State private statuses: ModelDownloadStatus[] = [];
  @State private provider: string = PROVIDER_MODEL_SCOPE;
  @State private query: string = '';
  @State private selectedTag: string = '';
  @State private selectedVendor: string = '';
  @State private updateIds: string[] = [];
  @State private loading: boolean = false;
  @State private error: string = '';

  aboutToAppear(): void {
    this.load(false);
  }

  private load(refreshNetwork: boolean): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.loading = true;
    AppInitializer.initialize(context).then(() => {
      return ModelMarketService.getInstance().load(context, refreshNetwork);
    }).then((catalog: MarketCatalogRecord) => {
      this.catalog = catalog;
      this.applyFilters();
      return this.loadStatuses();
    }).then(() => {
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.loading = false;
    });
  }

  private applyFilters(): void {
    const query = this.query.trim().toLowerCase();
    this.models = this.catalog.models.filter((model: MarketModelRecord) => {
      if (!ProviderAdapters.repository(model, this.provider)) {
        return false;
      }
      if (this.selectedTag && model.tags.indexOf(this.selectedTag) < 0) {
        return false;
      }
      if (this.selectedVendor && model.vendor !== this.selectedVendor) {
        return false;
      }
      return !query || model.modelName.toLowerCase().indexOf(query) >= 0 ||
        model.vendor.toLowerCase().indexOf(query) >= 0 ||
        model.tags.some((tag: string) => tag.toLowerCase().indexOf(query) >= 0);
    });
  }

  private async loadStatuses(): Promise<void> {
    const tasks = await DownloadRepository.getInstance().list();
    const statuses: ModelDownloadStatus[] = [];
    tasks.forEach((task: DownloadTaskRecord) => {
      let status = statuses.find((item: ModelDownloadStatus) => item.modelId === task.modelId);
      if (!status) {
        status = new ModelDownloadStatus();
        status.modelId = task.modelId;
        status.state = DownloadState.COMPLETED;
        statuses.push(status);
      }
      status.downloadedBytes += task.downloadedBytes;
      status.totalBytes += task.totalBytes;
      if (task.state === DownloadState.FAILED) {
        status.state = DownloadState.FAILED;
      } else if (task.state === DownloadState.DOWNLOADING && status.state !== DownloadState.FAILED) {
        status.state = DownloadState.DOWNLOADING;
      } else if (task.state === DownloadState.PAUSED && status.state === DownloadState.COMPLETED) {
        status.state = DownloadState.PAUSED;
      } else if (task.state === DownloadState.PENDING && status.state === DownloadState.COMPLETED) {
        status.state = DownloadState.PENDING;
      }
    });
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    for (let index = 0; index < statuses.length; index++) {
      const status = statuses[index];
      if ((status.state === DownloadState.DOWNLOADING || status.state === DownloadState.PENDING) &&
        !DownloadManager.getInstance().isActive(status.modelId)) {
        status.state = DownloadState.PAUSED;
      }
      if (status.state === DownloadState.COMPLETED) {
        const task = tasks.find((item: DownloadTaskRecord) => item.modelId === status.modelId);
        const marker = '/.mnn-downloads/';
        const markerIndex = task ? task.targetPath.indexOf(marker) : -1;
        if (task && markerIndex >= 0) {
          const relativePath = task.targetPath.substring(markerIndex + marker.length);
          const separator = relativePath.indexOf('/');
          const directoryName = separator > 0 ? relativePath.substring(0, separator) : '';
          if (!directoryName || !fs.accessSync(`${context.filesDir}/.mnnmodels/${directoryName}/config.json`)) {
            status.state = DownloadState.PENDING;
            await DownloadRepository.getInstance().removeModel(status.modelId);
          }
        }
      }
    }
    this.statuses = statuses;
    this.computeUpdates();
  }

  /**
   * Flags installed models whose local file set no longer matches the catalog size. This mirrors
   * the Android "update available" badge without requiring a network round trip per card.
   */
  private computeUpdates(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const ids: string[] = [];
    this.models.forEach((model: MarketModelRecord) => {
      if (model.fileSize <= 0) {
        return;
      }
      const directory = `${context.filesDir}/.mnnmodels/${this.safeName(model.modelName)}`;
      if (!fs.accessSync(directory)) {
        return;
      }
      const localBytes = this.directorySize(directory);
      const tolerance = Math.max(1024 * 1024, model.fileSize * 0.005);
      if (localBytes > 0 && Math.abs(localBytes - model.fileSize) > tolerance) {
        ids.push(this.modelId(model));
      }
    });
    this.updateIds = ids;
  }

  private directorySize(path: string): number {
    try {
      let total = 0;
      const names = fs.listFileSync(path);
      for (let index = 0; index < names.length; index++) {
        const child = `${path}/${names[index]}`;
        const stat = fs.lstatSync(child);
        if (stat.isSymbolicLink()) {
          continue;
        }
        total += stat.isDirectory() ? this.directorySize(child) : stat.size;
      }
      return total;
    } catch (error) {
      return 0;
    }
  }

  private safeName(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, '-').replace(/^-+|-+$/g, '').substring(0, 64);
  }

  private isInstalled(model: MarketModelRecord): boolean {
    if (this.statusFor(model).state === DownloadState.COMPLETED) {
      return true;
    }
    // Models imported from the file picker have no download records; treat their directory as
    // installed so update detection applies to them too.
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    return fs.accessSync(`${context.filesDir}/.mnnmodels/${this.safeName(model.modelName)}`);
  }

  private isUpdateAvailable(model: MarketModelRecord): boolean {
    return this.updateIds.indexOf(this.modelId(model)) >= 0;
  }

  private modelId(model: MarketModelRecord): string {
    return `${this.provider}/${ProviderAdapters.repository(model, this.provider)}`;
  }

  private statusFor(model: MarketModelRecord): ModelDownloadStatus {
    const modelId = this.modelId(model);
    const status = this.statuses.find((item: ModelDownloadStatus) => item.modelId === modelId);
    if (status) {
      return status;
    }
    const empty = new ModelDownloadStatus();
    empty.modelId = modelId;
    return empty;
  }

  private updateStatus(status: ModelDownloadStatus): void {
    const next = this.statuses.filter((item: ModelDownloadStatus) => item.modelId !== status.modelId);
    next.push(status);
    this.statuses = next;
    if (status.state === DownloadState.COMPLETED) {
      this.computeUpdates();
    }
  }

  private startDownload(model: MarketModelRecord, replace: boolean = false): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.error = '';
    DownloadManager.getInstance().start(context, model, this.provider,
      (status: ModelDownloadStatus) => this.updateStatus(status), replace).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private pauseDownload(model: MarketModelRecord): void {
    DownloadManager.getInstance().pause(this.modelId(model)).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private resumeDownload(model: MarketModelRecord): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    DownloadManager.getInstance().resume(context, model, this.provider,
      (status: ModelDownloadStatus) => this.updateStatus(status)).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private cancelDownload(model: MarketModelRecord): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    DownloadManager.getInstance().cancel(context, this.modelId(model)).then(() => {
      const status = new ModelDownloadStatus();
      status.modelId = this.modelId(model);
      this.updateStatus(status);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private retryDownload(model: MarketModelRecord): void {
    const modelId = this.modelId(model);
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    DownloadManager.getInstance().cancel(context, modelId).then(() => {
      this.startDownload(model);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private formatBytes(bytes: number): string {
    if (bytes <= 0) {
      return '未知大小';
    }
    if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  private progressText(status: ModelDownloadStatus): string {
    if (status.state === DownloadState.DOWNLOADING && status.totalBytes > 0) {
      return `${(status.downloadedBytes * 100 / status.totalBytes).toFixed(1)}%`;
    }
    if (status.state === DownloadState.PAUSED) {
      return '已暂停';
    }
    if (status.state === DownloadState.FAILED) {
      return status.error || '失败';
    }
    if (status.state === DownloadState.COMPLETED) {
      return '已安装';
    }
    return '';
  }

  private vendorOptions(): string[] {
    const vendors: string[] = [];
    this.catalog.vendorOrder.forEach((vendor: string) => {
      if (vendor && vendors.indexOf(vendor) < 0) {
        vendors.push(vendor);
      }
    });
    this.catalog.models.forEach((model: MarketModelRecord) => {
      if (model.vendor && vendors.indexOf(model.vendor) < 0) {
        vendors.push(model.vendor);
      }
    });
    return vendors;
  }

  private openDetail(model: MarketModelRecord): void {
    let controller: CustomDialogController | null = null;
    controller = new CustomDialogController({
      builder: MarketDetailSheet({
        model: model,
        provider: this.provider,
        installed: this.isInstalled(model),
        updateAvailable: this.isUpdateAvailable(model),
        onDownload: () => this.startDownload(model, this.isUpdateAvailable(model))
      }),
      autoCancel: true,
      alignment: DialogAlignment.Bottom,
      customStyle: true
    });
    controller.open();
  }

  @Builder
  private downloadAction(model: MarketModelRecord) {
    if (this.statusFor(model).state === DownloadState.DOWNLOADING) {
      Button('暂停')
        .height(34)
        .fontSize(12)
        .onClick(() => this.pauseDownload(model))
    } else if (this.statusFor(model).state === DownloadState.PAUSED) {
      Button('继续')
        .height(34)
        .fontSize(12)
        .onClick(() => this.resumeDownload(model))
    } else if (this.statusFor(model).state === DownloadState.FAILED) {
      Button('重试')
        .height(34)
        .fontSize(12)
        .onClick(() => this.retryDownload(model))
    } else if (this.isInstalled(model)) {
      if (this.isUpdateAvailable(model)) {
        Button('更新')
          .height(34)
          .fontSize(12)
          .backgroundColor('#FFF4E5')
          .fontColor('#B26A00')
          .onClick(() => this.startDownload(model, true))
      } else {
        Button('已安装')
          .height(34)
          .fontSize(12)
          .enabled(false)
      }
    } else {
      Button('下载')
        .height(34)
        .fontSize(12)
        .onClick(() => this.startDownload(model))
    }
  }

  build() {
    Column() {
      Row({ space: 8 }) {
        Text('模型市场')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .layoutWeight(1)
        Button('刷新')
          .height(34)
          .fontSize(12)
          .enabled(!this.loading)
          .onClick(() => this.load(true))
      }
      .width('100%')
      .padding({ left: 18, right: 18, top: 18, bottom: 10 })

      Row({ space: 8 }) {
        ForEach([PROVIDER_MODEL_SCOPE, PROVIDER_HUGGING_FACE, PROVIDER_MODELERS], (item: string) => {
          Button(item)
            .height(32)
            .fontSize(11)
            .fontColor(this.provider === item ? Color.White : '#2864DC')
            .backgroundColor(this.provider === item ? '#2864DC' : '#EEF3FF')
            .onClick(() => {
              this.provider = item;
              this.applyFilters();
              this.loadStatuses();
            })
        })
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 10 })

      TextInput({ text: this.query, placeholder: '搜索模型、厂商或标签' })
        .margin({ left: 16, right: 16, bottom: 10 })
        .onChange((value: string) => {
          this.query = value;
          this.applyFilters();
        })

      Scroll() {
        Row({ space: 8 }) {
          Button('全部')
            .height(30)
            .fontSize(11)
            .fontColor(this.selectedTag ? '#2864DC' : Color.White)
            .backgroundColor(this.selectedTag ? '#EEF3FF' : '#2864DC')
            .onClick(() => {
              this.selectedTag = '';
              this.applyFilters();
            })
          ForEach(this.catalog.quickFilterTags, (tag: string) => {
            Button(tag)
              .height(30)
              .fontSize(11)
              .fontColor(this.selectedTag === tag ? Color.White : '#2864DC')
              .backgroundColor(this.selectedTag === tag ? '#2864DC' : '#EEF3FF')
              .onClick(() => {
                this.selectedTag = tag;
                this.applyFilters();
              })
          })
        }
      }
      .scrollable(ScrollDirection.Horizontal)
      .scrollBar(BarState.Off)
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 10 })

      Scroll() {
        Row({ space: 8 }) {
          Button('全部厂商')
            .height(30)
            .fontSize(11)
            .fontColor(this.selectedVendor ? '#2864DC' : Color.White)
            .backgroundColor(this.selectedVendor ? '#EEF3FF' : '#2864DC')
            .onClick(() => {
              this.selectedVendor = '';
              this.applyFilters();
            })
          ForEach(this.vendorOptions(), (vendor: string) => {
            Button(vendor)
              .height(30)
              .fontSize(11)
              .fontColor(this.selectedVendor === vendor ? Color.White : '#2864DC')
              .backgroundColor(this.selectedVendor === vendor ? '#2864DC' : '#EEF3FF')
              .onClick(() => {
                this.selectedVendor = vendor;
                this.applyFilters();
              })
          }, (vendor: string) => vendor)
        }
      }
      .scrollable(ScrollDirection.Horizontal)
      .scrollBar(BarState.Off)
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 10 })

      List({ space: 10 }) {
        ForEach(this.models, (model: MarketModelRecord) => {
          ListItem() {
            Column({ space: 7 }) {
              Row({ space: 8 }) {
                Text(model.modelName)
                  .fontSize(16)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(model.vendor)
                  .fontSize(11)
                  .fontColor('#6B7280')
              }
              if (model.description) {
                Text(model.description)
                  .fontSize(12)
                  .fontColor('#5F6670')
                  .maxLines(2)
              }
              Text(`${model.sizeB}B · ${this.formatBytes(model.fileSize)} · ${model.tags.join(' · ')}`)
                .fontSize(11)
                .fontColor('#7A7F87')
                .maxLines(2)
              Row({ space: 8 }) {
                Text(this.progressText(this.statusFor(model)))
                  .fontSize(12)
                  .fontColor('#2864DC')
                  .layoutWeight(1)
                if (this.statusFor(model).state === DownloadState.DOWNLOADING ||
                  this.statusFor(model).state === DownloadState.PAUSED ||
                  this.statusFor(model).state === DownloadState.FAILED) {
                  Button('取消')
                    .height(34)
                    .fontSize(12)
                    .backgroundColor('#FDECEC')
                    .fontColor('#C62828')
                    .onClick(() => this.cancelDownload(model))
                }
                this.downloadAction(model)
              }
            }
            .alignItems(HorizontalAlign.Start)
            .padding(14)
            .borderRadius(12)
            .backgroundColor(Color.White)
            .onClick(() => this.openDetail(model))
          }
        }, (model: MarketModelRecord) => `${this.provider}-${model.modelName}`)
      }
      .layoutWeight(1)
      .padding({ left: 16, right: 16, bottom: 12 })

      if (this.loading) {
        Text('正在加载模型市场…')
          .fontSize(13)
          .fontColor('#6B7280')
          .padding(12)
      }
      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .padding(12)
      }
      Text('下载会在后台继续执行（系统可能显示“数据传输”通知）；私有仓库可在“设置”中填写平台令牌。')
        .fontSize(11)
        .fontColor('#7A7F87')
        .padding({ left: 16, right: 16, bottom: 10 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}
