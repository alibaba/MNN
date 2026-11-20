import { tr } from '../services/I18n';
@Component
export struct ReasoningPanel {
  @Prop reasoning: string;
  @Prop isThinking: boolean;
  @State private expanded: boolean = true;

  build() {
    // Android shows reasoning as a "Thinking..." label plus a left-bordered indented block.
    Column({ space: 6 }) {
      Text(this.isThinking ? tr('Thinking...') : tr('思考过程'))
        .fontSize(13)
        .fontColor($r('app.color.text_secondary'))
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
          .padding({ left: 12 })
          .border({ width: { left: 3 }, color: $r('app.color.divider') })
          .copyOption(CopyOptions.LocalDevice)
          .onClick(() => {
            this.expanded = !this.expanded;
          })
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }
}
