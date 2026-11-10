import { PRIVACY_POLICY_TEXT } from '../services/PrivacyPolicy';

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
      Text('隐私政策')
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor('#171A1F')
        .width('100%')
      Scroll() {
        Text(PRIVACY_POLICY_TEXT)
          .fontSize(14)
          .lineHeight(22)
          .fontColor('#303640')
          .width('100%')
      }
      .height(330)
      .width('100%')
      .scrollBar(BarState.Auto)

      Row({ space: 10 }) {
        Button('不同意并退出')
          .layoutWeight(1)
          .height(42)
          .fontSize(14)
          .fontColor('#5F6670')
          .backgroundColor('#F1F3F5')
          .onClick(() => {
            this.controller?.close();
            this.onDecline();
          })
        Button('同意并继续')
          .layoutWeight(1)
          .height(42)
          .fontSize(14)
          .fontColor(Color.White)
          .backgroundColor('#2864DC')
          .onClick(() => {
            this.controller?.close();
            this.onAgree();
          })
      }
      .width('100%')
    }
    .width('100%')
    .padding(20)
    .backgroundColor(Color.White)
    .borderRadius(16)
  }
}