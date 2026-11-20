import { Nav } from '@taobao-ohos/nav'

@Entry
@Component
export default struct DemoPage {
  @State data?: object = undefined;
  @State bizDataString: string = "";

  aboutToAppear(): void {
    this.data = Nav.getPageParams(this);
    this.bizDataString = this.data['bizData'];
  }

  build() {
    Column() {
      Text("DemoPage")
        .fontSize('10%')
      Line()
      Text("透传数据：" + this.bizDataString)
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
}