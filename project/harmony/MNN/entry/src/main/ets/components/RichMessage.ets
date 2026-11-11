import { MarkdownBlock, MarkdownBlockType, MarkdownParser } from '../services/MarkdownParser';

@Component
export struct RichMessage {
  @Prop content: string;

  build() {
    Column({ space: 8 }) {
      ForEach(MarkdownParser.parse(this.content), (block: MarkdownBlock) => {
        if (block.type === MarkdownBlockType.HEADING) {
          Text(block.content)
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .fontColor($r('app.color.text_primary_alt'))
            .width('100%')
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.CODE) {
          Scroll() {
            Text(block.content)
              .fontSize(13)
              .fontFamily('monospace')
              .fontColor($r('app.color.code_text'))
              .copyOption(CopyOptions.LocalDevice)
          }
          .scrollable(ScrollDirection.Horizontal)
          .width('100%')
          .padding(10)
          .borderRadius(8)
          .backgroundColor($r('app.color.code_bg'))
        } else if (block.type === MarkdownBlockType.QUOTE) {
          Text(block.content)
            .fontSize(15)
            .lineHeight(22)
            .fontColor($r('app.color.quote_text'))
            .width('100%')
            .padding({ left: 10 })
            .border({ width: { left: 3 }, color: $r('app.color.quote_border') })
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.TABLE) {
          Text(block.content)
            .fontSize(13)
            .fontFamily('monospace')
            .lineHeight(21)
            .fontColor($r('app.color.text_body_alt'))
            .width('100%')
            .padding(8)
            .borderRadius(8)
            .backgroundColor($r('app.color.divider_soft'))
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.MATH) {
          Text(block.content)
            .fontSize(15)
            .fontFamily('serif')
            .fontColor($r('app.color.link'))
            .width('100%')
            .textAlign(TextAlign.Center)
            .padding(10)
            .borderRadius(8)
            .backgroundColor($r('app.color.brand_tint_alt'))
            .copyOption(CopyOptions.LocalDevice)
        } else {
          Text(block.content)
            .fontSize(16)
            .lineHeight(24)
            .fontColor($r('app.color.text_primary_alt'))
            .width('100%')
            .copyOption(CopyOptions.LocalDevice)
        }
      }, (block: MarkdownBlock) => `${block.id}:${block.content.length}`)
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
  }
}
