import { tr } from '../services/I18n';
import { common, Want } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
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
  updateAvailable: boolean = false;
  onDownload: () => void = () => {};

  private openWebPage(): void {
    const url = ProviderAdapters.webPage(this.model, this.provider);
    if (!url) {
      promptAction.showToast({ message: tr('该模型没有可访问的网页') });
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    try {
      const want: Want = {
        action: 'ohos.want.action.viewData',
        entities: ['entity.system.browsable'],
        uri: url
      };
      context.startAbility(want).catch(() => {
        promptAction.showToast({ message: tr('无法打开浏览器') });
      });
    } catch (error) {
      promptAction.showToast({ message: tr('无法打开浏览器') });
    }
  }

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
        .fontColor($r('app.color.text_primary'))
        .width('100%')
      Scroll() {
        Column({ space: 8 }) {
          Text(`${tr('厂商：')}${this.model.vendor || tr('未知')}`)
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
            .width('100%')
          Text(`${tr('参数：')}${this.model.sizeB}${tr('B · 文件：')}${this.formatBytes(this.model.fileSize)}`)
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
            .width('100%')
          if (this.model.tags.length > 0) {
            Text(`${tr('标签：')}${this.model.tags.join(' · ')}`)
              .fontSize(14)
              .fontColor($r('app.color.text_body'))
              .width('100%')
          }
          if (this.model.description) {
            Text(this.model.description)
              .fontSize(14)
              .lineHeight(21)
              .fontColor($r('app.color.text_body_alt'))
              .width('100%')
          }
          if (this.updateAvailable) {
            Text(tr('本地文件与仓库不一致，可下载最新版本'))
              .fontSize(12)
              .fontColor($r('app.color.warning'))
              .width('100%')
          }
          Text(`${tr('来源：')}${this.provider}/${ProviderAdapters.repository(this.model, this.provider)}`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .width('100%')
          Button(tr('在浏览器打开模型卡片'))
            .fontSize(12)
            .fontColor($r('app.color.brand'))
            .backgroundColor(Color.Transparent)
            .height(30)
            .padding(0)
            .alignSelf(ItemAlign.Start)
            .onClick(() => this.openWebPage())
        }
        .alignItems(HorizontalAlign.Start)
        .width('100%')
      }
      .layoutWeight(1)
      .width('100%')

      Row({ space: 12 }) {
        Button(this.installed && !this.updateAvailable ? tr('已安装') : (this.updateAvailable ? tr('更新') : tr('下载')))
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .enabled(!this.installed || this.updateAvailable)
          .backgroundColor(this.installed && !this.updateAvailable ? $r('app.color.surface_muted') : $r('app.color.brand'))
          .fontColor(this.installed && !this.updateAvailable ? $r('app.color.text_muted') : Color.White)
          .onClick(() => {
            this.controller?.close();
            this.onDownload();
          })
        Button(tr('关闭'))
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .backgroundColor($r('app.color.surface_muted'))
          .fontColor($r('app.color.text_muted'))
          .onClick(() => this.controller?.close())
      }
      .width('100%')
    }
    .width('100%')
    .height('55%')
    .padding(20)
    .backgroundColor($r('app.color.surface'))
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}