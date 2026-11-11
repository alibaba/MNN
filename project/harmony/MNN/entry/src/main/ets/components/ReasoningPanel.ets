import { tr } from '../services/I18n';
@Component
export struct ReasoningPanel {
  @Prop reasoning: string;
  @Prop isThinking: boolean;
  @State private expanded: boolean = false;

  build() {
    Column({ space: 8 }) {
      Row() {
        Text(this.isThinking ? tr('正在思考') : tr('思考过程'))
          .fontSize(13)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_muted'))
          .layoutWeight(1)
        Text(this.expanded ? tr('收起') : tr('展开'))
          .fontSize(12)
          .fontColor($r('app.color.brand'))
      }
      .width('100%')
      .onClick(() => {
        this.expanded = !this.expanded;
      })

      if (this.expanded) {
        Text(this.reasoning)
          .fontSize(14)
          .lineHeight(21)
          .fontColor($r('app.color.text_hint'))
          .width('100%')
          .copyOption(CopyOptions.LocalDevice)
      }
    }
    .width('100%')
    .padding(10)
    .borderRadius(10)
    .backgroundColor($r('app.color.divider_soft'))
  }
}
