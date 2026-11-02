@Component
export struct ReasoningPanel {
  @Prop reasoning: string;
  @Prop isThinking: boolean;
  @State private expanded: boolean = false;

  build() {
    Column({ space: 8 }) {
      Row() {
        Text(this.isThinking ? '正在思考' : '思考过程')
          .fontSize(13)
          .fontWeight(FontWeight.Medium)
          .fontColor('#5F6670')
          .layoutWeight(1)
        Text(this.expanded ? '收起' : '展开')
          .fontSize(12)
          .fontColor('#2864DC')
      }
      .width('100%')
      .onClick(() => {
        this.expanded = !this.expanded;
      })

      if (this.expanded) {
        Text(this.reasoning)
          .fontSize(14)
          .lineHeight(21)
          .fontColor('#606771')
          .width('100%')
          .copyOption(CopyOptions.LocalDevice)
      }
    }
    .width('100%')
    .padding(10)
    .borderRadius(10)
    .backgroundColor('#E9ECF1')
  }
}
