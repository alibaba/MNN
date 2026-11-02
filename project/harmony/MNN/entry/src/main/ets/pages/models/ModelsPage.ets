import { common } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import { ModelRepository } from '../../data/ModelRepository';
import { ModelRecord, ModelState, ModelStorageStats } from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../../services/SettingsStore';

@Component
export struct ModelsPage {
  @Link activeModelId: string;
  @Link selectedTab: number;
  @State private models: ModelRecord[] = [];
  @State private storage: ModelStorageStats = new ModelStorageStats();
  @State private query: string = '';
  @State private error: string = '';
  @State private busy: boolean = false;

  aboutToAppear(): void {
    this.refresh(true);
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
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private visibleModels(): ModelRecord[] {
    const query = this.query.trim().toLowerCase();
    if (!query) {
      return this.models;
    }
    return this.models.filter((model: ModelRecord) =>
      model.name.toLowerCase().indexOf(query) >= 0 || model.configPath.toLowerCase().indexOf(query) >= 0);
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
      this.selectedTab = 0;
    }).catch((error: Error) => {
      this.error = error.message;
    });
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
      title: '删除模型',
      message: `确认删除 ${model.name}？此操作会移除受管模型文件。`,
      primaryButton: {
        value: '取消',
        action: () => {
        }
      },
      secondaryButton: {
        value: '删除',
        fontColor: '#C62828',
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

  private formatBytes(bytes: number): string {
    if (bytes < 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  build() {
    Column() {
      Row({ space: 8 }) {
        Text('本地模型')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .layoutWeight(1)
        Button('导入')
          .height(36)
          .fontSize(13)
          .enabled(!this.busy)
          .onClick(() => this.importModel())
        Button('扫描')
          .height(36)
          .fontSize(13)
          .enabled(!this.busy)
          .onClick(() => this.refresh(true))
      }
      .width('100%')
      .padding(18)

      TextInput({ text: this.query, placeholder: '搜索名称或路径' })
        .margin({ left: 16, right: 16, bottom: 10 })
        .onChange((value: string) => {
          this.query = value;
        })

      Text(`模型 ${this.formatBytes(this.storage.modelBytes)} · 可用 ${this.formatBytes(this.storage.freeBytes)} / ` +
        `${this.formatBytes(this.storage.totalBytes)}`)
        .fontSize(12)
        .fontColor('#6B7280')
        .width('100%')
        .padding({ left: 18, right: 18, bottom: 10 })

      List({ space: 12 }) {
        ForEach(this.visibleModels(), (model: ModelRecord) => {
          ListItem() {
            Column({ space: 8 }) {
              Row() {
                Text(model.name)
                  .fontSize(18)
                  .fontWeight(FontWeight.Medium)
                  .layoutWeight(1)
                Text(model.state === ModelState.READY ? '可用' : '配置无效')
                  .fontSize(12)
                  .fontColor(model.state === ModelState.READY ? '#168653' : '#C62828')
                  .backgroundColor(model.state === ModelState.READY ? '#E9F8F0' : '#FDECEC')
                  .padding({ left: 9, right: 9, top: 4, bottom: 4 })
                  .borderRadius(10)
              }
              Text(model.configPath)
                .fontSize(12)
                .fontColor('#6B7280')
                .maxLines(2)
              Text(this.formatBytes(model.sizeBytes))
                .fontSize(12)
                .fontColor('#6B7280')
              if (model.isProtected) {
                Text('内置沙箱模型 · 受保护，不会被移动或清理')
                  .fontSize(12)
                  .fontColor('#2864DC')
              }
              Row({ space: 8 }) {
                Button(model.isPinned ? '取消置顶' : '置顶')
                  .height(32)
                  .fontSize(12)
                  .enabled(!this.busy)
                  .onClick(() => this.togglePinned(model))
                Button(this.activeModelId === model.id ? '使用中' : '使用')
                  .height(32)
                  .fontSize(12)
                  .enabled(!this.busy && model.state === ModelState.READY && this.activeModelId !== model.id)
                  .onClick(() => this.useModel(model))
                Button('删除')
                  .height(32)
                  .fontSize(12)
                  .enabled(!this.busy && !model.isProtected && !model.isLegacy && this.activeModelId !== model.id)
                  .backgroundColor('#FDECEC')
                  .fontColor('#C62828')
                  .onClick(() => this.confirmDelete(model))
              }
            }
            .alignItems(HorizontalAlign.Start)
            .padding(16)
            .borderRadius(14)
            .backgroundColor(Color.White)
          }
        }, (model: ModelRecord) => model.id)
      }
      .layoutWeight(1)
      .padding({ left: 16, right: 16, bottom: 16 })

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .padding(16)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}
