import { tr } from '../services/I18n';
/**
 * Scrollable read-only text sheet with a copy action, used for the model config info dialog.
 */
@CustomDialog
export struct InfoSheet {
  controller?: CustomDialogController;
  title: string = '';
  message: string = '';
  onCopy: () => void = () => {};

  build() {
    Column({ space: 12 }) {
      Text(this.title)
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
      Scroll() {
        Text(this.message)
          .fontSize(13)
          .fontColor($r('app.color.text_body_alt'))
          .fontFamily('monospace')
          .copyOption(CopyOptions.LocalDevice)
          .width('100%')
      }
      .layoutWeight(1)
      .width('100%')
      .padding(12)
      .borderRadius(10)
      .backgroundColor($r('app.color.surface_alt'))

      Row({ space: 12 }) {
        Button(tr('复制'))
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .backgroundColor($r('app.color.brand_tint'))
          .fontColor($r('app.color.brand'))
          .onClick(() => this.onCopy())
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