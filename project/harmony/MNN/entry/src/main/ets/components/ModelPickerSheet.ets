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
        .fontColor('#171A1F')
        .width('100%')
      List({ space: 8 }) {
        ForEach(this.models, (model: ModelRecord) => {
          ListItem() {
            Row({ space: 10 }) {
              Column({ space: 2 }) {
                Text(model.name)
                  .fontSize(16)
                  .fontColor('#171A1F')
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                Text(`${(model.sizeBytes / 1024 / 1024).toFixed(1)} MB`)
                  .fontSize(12)
                  .fontColor('#6B7280')
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
              if (model.id === this.activeModelId) {
                Text('使用中')
                  .fontSize(12)
                  .fontColor('#2864DC')
              }
            }
            .width('100%')
            .padding(12)
            .borderRadius(10)
            .backgroundColor(model.id === this.activeModelId ? '#EEF3FF' : '#F6F7F9')
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
        .fontColor('#5F6670')
        .backgroundColor('#F1F3F5')
        .onClick(() => this.controller?.close())
    }
    .width('100%')
    .height('60%')
    .padding(20)
    .backgroundColor(Color.White)
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}