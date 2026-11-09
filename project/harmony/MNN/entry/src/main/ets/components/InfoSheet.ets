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
        .fontColor('#171A1F')
        .width('100%')
      Scroll() {
        Text(this.message)
          .fontSize(13)
          .fontColor('#303640')
          .fontFamily('monospace')
          .copyOption(CopyOptions.LocalDevice)
          .width('100%')
      }
      .layoutWeight(1)
      .width('100%')
      .padding(12)
      .borderRadius(10)
      .backgroundColor('#F6F7F9')

      Row({ space: 12 }) {
        Button('复制')
          .layoutWeight(1)
          .height(40)
          .fontSize(15)
          .backgroundColor('#EEF3FF')
          .fontColor('#2864DC')
          .onClick(() => this.onCopy())
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