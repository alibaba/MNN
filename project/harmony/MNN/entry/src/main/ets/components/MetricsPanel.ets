import { tr } from '../services/I18n';
import { GenerationMetrics, GenerationState } from '../domain/AppTypes';

@Component
export struct MetricsPanel {
  @Prop metrics: GenerationMetrics;
  @Prop state: GenerationState;
  @State private expanded: boolean = false;

  build() {
    if (this.metrics.totalMs > 0 || this.state === GenerationState.CANCELED) {
      Column({ space: 6 }) {
        Row() {
          Text(this.state === GenerationState.CANCELED ? tr('生成已停止') : tr('性能指标'))
            .fontSize(12)
            .fontColor($r('app.color.text_secondary_alt'))
            .layoutWeight(1)
          Text(this.expanded ? tr('收起') : tr('详情'))
            .fontSize(12)
            .fontColor($r('app.color.brand'))
        }
        .width('100%')
        .onClick(() => {
          this.expanded = !this.expanded;
        })

        if (this.expanded) {
          Text(`${tr('加载 ')}${(this.metrics.loadMs / 1000).toFixed(2)}${tr('s · 首字 ')}${(this.metrics.firstTokenMs / 1000).toFixed(2)}s`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary_alt'))
          Text(`Prompt ${this.metrics.promptTokens} tokens / ${(this.metrics.prefillMs / 1000).toFixed(2)}s`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary_alt'))
          Text(`Decode ${this.metrics.generatedTokens} tokens / ${(this.metrics.decodeMs / 1000).toFixed(2)}s`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary_alt'))
          Text(`${tr('总耗时 ')}${(this.metrics.totalMs / 1000).toFixed(2)}s`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary_alt'))
        }
      }
      .width('100%')
      .padding({ top: 8 })
    }
  }
}
