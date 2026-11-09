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
            .fontColor('#1F2329')
            .width('100%')
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.CODE) {
          Scroll() {
            Text(block.content)
              .fontSize(13)
              .fontFamily('monospace')
              .fontColor('#E7EAF0')
              .copyOption(CopyOptions.LocalDevice)
          }
          .scrollable(ScrollDirection.Horizontal)
          .width('100%')
          .padding(10)
          .borderRadius(8)
          .backgroundColor('#252A34')
        } else if (block.type === MarkdownBlockType.QUOTE) {
          Text(block.content)
            .fontSize(15)
            .lineHeight(22)
            .fontColor('#555D68')
            .width('100%')
            .padding({ left: 10 })
            .border({ width: { left: 3 }, color: '#8AAAF0' })
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.TABLE) {
          Text(block.content)
            .fontSize(13)
            .fontFamily('monospace')
            .lineHeight(21)
            .fontColor('#303640')
            .width('100%')
            .padding(8)
            .borderRadius(8)
            .backgroundColor('#E9ECF1')
            .copyOption(CopyOptions.LocalDevice)
        } else if (block.type === MarkdownBlockType.MATH) {
          Text(block.content)
            .fontSize(15)
            .fontFamily('serif')
            .fontColor('#273B68')
            .width('100%')
            .textAlign(TextAlign.Center)
            .padding(10)
            .borderRadius(8)
            .backgroundColor('#EDF3FF')
            .copyOption(CopyOptions.LocalDevice)
        } else {
          Text(block.content)
            .fontSize(16)
            .lineHeight(24)
            .fontColor('#1F2329')
            .width('100%')
            .copyOption(CopyOptions.LocalDevice)
        }
      }, (block: MarkdownBlock) => `${block.id}:${block.content.length}`)
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
  }
}
