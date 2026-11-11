import { common } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import { ImageActions } from '../services/ImageActions';

/**
 * Full-screen image viewer for chat attachments. Tapping the image closes it; the bottom bar
 * offers the same save / share actions as the Android client's generated-image menu.
 */
@CustomDialog
export struct ImagePreviewDialog {
  controller?: CustomDialogController;
  path: string = '';
  @State private busy: boolean = false;

  private context(): common.UIAbilityContext {
    return this.getUIContext().getHostContext() as common.UIAbilityContext;
  }

  private save(): void {
    if (this.busy) {
      return;
    }
    this.busy = true;
    ImageActions.saveToGallery(this.context(), this.path).then(() => {
      promptAction.showToast({ message: '已保存到相册' });
    }).catch((error: Error) => {
      promptAction.showToast({ message: `保存失败：${error.message}` });
    }).finally(() => {
      this.busy = false;
    });
  }

  private share(): void {
    if (this.busy) {
      return;
    }
    this.busy = true;
    ImageActions.share(this.context(), this.path).then(() => {
    }).catch((error: Error) => {
      promptAction.showToast({ message: `分享失败：${error.message}` });
    }).finally(() => {
      this.busy = false;
    });
  }

  build() {
    Column() {
      Image(`file://${this.path}`)
        .objectFit(ImageFit.Contain)
        .width('100%')
        .layoutWeight(1)
        .onClick(() => this.controller?.close())

      Row({ space: 10 }) {
        Button('保存图片')
          .layoutWeight(1)
          .height(42)
          .fontSize(13)
          .padding({ left: 0, right: 0 })
          .fontColor(Color.White)
          .backgroundColor($r('app.color.preview_button'))
          .enabled(!this.busy)
          .onClick(() => this.save())
        Button('分享')
          .layoutWeight(1)
          .height(42)
          .fontSize(13)
          .padding({ left: 0, right: 0 })
          .fontColor(Color.White)
          .backgroundColor($r('app.color.brand'))
          .enabled(!this.busy)
          .onClick(() => this.share())
        Button('关闭')
          .layoutWeight(1)
          .height(42)
          .fontSize(13)
          .padding({ left: 0, right: 0 })
          .fontColor(Color.White)
          .backgroundColor($r('app.color.preview_button'))
          .onClick(() => this.controller?.close())
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 12, bottom: 24 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}