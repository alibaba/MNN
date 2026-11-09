import { MarketModelRecord } from '../domain/AppTypes';
import { ProviderAdapters } from '../services/ProviderAdapters';

/**
 * Model detail sheet for the market list: shows the same information the Android client displays
 * on a model card and lets the user start the download from there.
 */
@CustomDialog
export struct MarketDetailSheet {
  controller?: CustomDialogController;
  model: MarketModelRecord = new MarketModelRecord();
  provider: string = '';
  installed: boolean = false;
  onDownload: () => void = () => {};

  private formatBytes(bytes: number): string {
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  build() {
    Column({ space: 12 }) {
      Text(this.model.modelName)
        .fontSize(19)
        .fontWeight(FontWeight.Bold)
        .fontColor('#171A1F')
        .width('100%')
      Scroll() {
        Column({ space: 8 }) {
          Text(`厂商：${this.model.vendor || '未知'}`)
            .fontSize(14)
            .fontColor('#4B5563')
            .width('100%')
          Text(`参数：${this.model.sizeB}B · 文件：${this.formatBytes(this.model.fileSize)}`)
            .fontSize(14)
            .fontColor('#4B5563')
            .width('100%')
          if (this.model.tags.length > 0) {
            Text(`标签：${this.model.tags.join(' · ')}`)
              .fontSize(14)
              .fontColor('#4B5563')
              .width('100%')
          }
          if (this.model.description) {
            Text(this.model.description)
              .fontSize(14)
              .lineHeight(21)
              .fontColor('#303640')
              .width('100%')
          }
          Text(`来源：${this.provider}/${ProviderAdapters.repository(this.model, this.provider)}`)
            .fontSize(12)
            .fontColor('#6B7280')
            .width('100%')
        }
        .alignItems(HorizontalAlign.Start)
        .width('100%')
      }
      .layoutWeight(1)
      .width('100%')

      Row({ space: 12 }) {
        Button(this.installed ? '已安装' : '下载')
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .enabled(!this.installed)
          .backgroundColor(this.installed ? '#F1F3F5' : '#2864DC')
          .fontColor(this.installed ? '#5F6670' : Color.White)
          .onClick(() => {
            this.controller?.close();
            this.onDownload();
          })
        Button('关闭')
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .backgroundColor('#F1F3F5')
          .fontColor('#5F6670')
          .onClick(() => this.controller?.close())
      }
      .width('100%')
    }
    .width('100%')
    .height('55%')
    .padding(20)
    .backgroundColor(Color.White)
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}