import { ModelRecord } from '../domain/AppTypes';

/**
 * Bottom sheet listing the installed, ready models so the chat header can switch models without
 * leaving the conversation (parity with the Android model switcher).
 */
@CustomDialog
export struct ModelPickerSheet {
  controller?: CustomDialogController;
  models: ModelRecord[] = [];
  activeModelId: string = '';
  onPick: (model: ModelRecord) => void = () => {};

  build() {
    Column({ space: 12 }) {
      Text('选择模型')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
      List({ space: 8 }) {
        ForEach(this.models, (model: ModelRecord) => {
          ListItem() {
            Row({ space: 10 }) {
              Column({ space: 2 }) {
                Text(model.name)
                  .fontSize(16)
                  .fontColor($r('app.color.text_primary'))
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                Text(`${(model.sizeBytes / 1024 / 1024).toFixed(1)} MB`)
                  .fontSize(12)
                  .fontColor($r('app.color.text_secondary'))
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
              if (model.id === this.activeModelId) {
                Text('使用中')
                  .fontSize(12)
                  .fontColor($r('app.color.brand'))
              }
            }
            .width('100%')
            .padding(12)
            .borderRadius(10)
            .backgroundColor(model.id === this.activeModelId ? $r('app.color.brand_tint') : $r('app.color.surface_alt'))
            .onClick(() => {
              this.controller?.close();
              this.onPick(model);
            })
          }
        }, (model: ModelRecord) => model.id)
      }
      .layoutWeight(1)
      .width('100%')

      Button('取消')
        .width('100%')
        .height(40)
        .fontSize(15)
        .fontColor($r('app.color.text_muted'))
        .backgroundColor($r('app.color.surface_muted'))
        .onClick(() => this.controller?.close())
    }
    .width('100%')
    .height('60%')
    .padding(20)
    .backgroundColor($r('app.color.surface'))
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}