import { tr } from '../services/I18n';
import { privacyPolicyText } from '../services/PrivacyPolicy';

/**
 * First-launch privacy notice. The user must explicitly agree before the app becomes usable,
 * matching the Android client's PrivacyPolicyDialogFragment.
 */
@CustomDialog
export struct PrivacyPolicySheet {
  controller?: CustomDialogController;
  onAgree: () => void = () => {};
  onDecline: () => void = () => {};

  build() {
    Column({ space: 12 }) {
      Text(tr('隐私政策'))
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')
      Scroll() {
        Text(privacyPolicyText())
          .fontSize(14)
          .lineHeight(22)
          .fontColor($r('app.color.text_body_alt'))
          .width('100%')
      }
      .height(330)
      .width('100%')
      .scrollBar(BarState.Auto)

      Row({ space: 10 }) {
        Button(tr('不同意并退出'))
          .layoutWeight(1)
          .height(42)
          .fontSize(14)
          .fontColor($r('app.color.text_muted'))
          .backgroundColor($r('app.color.surface_muted'))
          .onClick(() => {
            this.controller?.close();
            this.onDecline();
          })
        Button(tr('同意并继续'))
          .layoutWeight(1)
          .height(42)
          .fontSize(14)
          .fontColor(Color.White)
          .backgroundColor($r('app.color.brand'))
          .onClick(() => {
            this.controller?.close();
            this.onAgree();
          })
      }
      .width('100%')
    }
    .width('100%')
    .padding(20)
    .backgroundColor($r('app.color.surface'))
    .borderRadius(16)
  }
}