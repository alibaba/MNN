/**
 * Full-screen image viewer for chat attachments. Tapping anywhere closes it.
 */
@CustomDialog
export struct ImagePreviewDialog {
  controller?: CustomDialogController;
  path: string = '';

  build() {
    Column() {
      Image(`file://${this.path}`)
        .objectFit(ImageFit.Contain)
        .width('100%')
        .height('100%')
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
    .onClick(() => this.controller?.close())
  }
}