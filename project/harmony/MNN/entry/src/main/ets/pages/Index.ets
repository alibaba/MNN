import { common } from '@kit.AbilityKit';
import { ApiConsolePage } from './api/ApiConsolePage';
import { BenchmarkPage } from './benchmark/BenchmarkPage';
import { ChatPage } from './chat/ChatPage';
import { HistoryPage } from './history/HistoryPage';
import { MarketPage } from './market/MarketPage';
import { ModelsPage } from './models/ModelsPage';
import { SettingsPage } from './settings/SettingsPage';
import { AppInitializer } from '../services/AppInitializer';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../services/SettingsStore';

@Entry
@Component
struct Index {
  @State private selectedTab: number = 0;
  @State private requestedSessionId: string = '';
  @State private activeModelId: string = 'legacy-model';
  @State private settingsRevision: number = 0;

  aboutToAppear(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(async () => {
      this.activeModelId = await SettingsStore.getInstance().getString(ACTIVE_MODEL_ID_KEY, 'legacy-model');
      this.settingsRevision = await SettingsStore.getInstance().getSettingsRevision();
    });
  }

  build() {
    Tabs({ barPosition: BarPosition.End, index: this.selectedTab }) {
      TabContent() {
        ChatPage({
          requestedSessionId: this.requestedSessionId,
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision
        })
      }
      .tabBar('聊天')

      TabContent() {
        HistoryPage({
          requestedSessionId: this.requestedSessionId,
          selectedTab: this.selectedTab
        })
      }
      .tabBar('历史')

      TabContent() {
        ModelsPage({
          activeModelId: this.activeModelId,
          selectedTab: this.selectedTab
        })
      }
      .tabBar('模型')

      TabContent() {
        MarketPage()
      }
      .tabBar('市场')

      TabContent() {
        SettingsPage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision
        })
      }
      .tabBar('设置')

      TabContent() {
        BenchmarkPage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision
        })
      }
      .tabBar('测试')

      TabContent() {
        ApiConsolePage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision
        })
      }
      .tabBar('API')
    }
    .barMode(BarMode.Scrollable)
    .barHeight(54)
    .onChange((index: number) => {
      this.selectedTab = index;
    })
    .width('100%')
    .height('100%')
  }
}
