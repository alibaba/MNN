import { tr } from '../services/I18n';
import { common } from '@kit.AbilityKit';
import { ApiConsolePage } from './api/ApiConsolePage';
import { BenchmarkPage } from './benchmark/BenchmarkPage';
import { ChatPage } from './chat/ChatPage';
import { HistoryPage } from './history/HistoryPage';
import { MarketPage } from './market/MarketPage';
import { ModelsPage } from './models/ModelsPage';
import { SettingsPage } from './settings/SettingsPage';
import { PrivacyPolicySheet } from '../components/PrivacyPolicySheet';
import { ModelPickerSheet } from '../components/ModelPickerSheet';
import { ModelRepository } from '../data/ModelRepository';
import { ModelRecord, ModelState } from '../domain/AppTypes';
import { AppInitializer } from '../services/AppInitializer';
import { PRIVACY_POLICY_KEY, PRIVACY_POLICY_VERSION } from '../services/PrivacyPolicy';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../services/SettingsStore';

const VIEW_MAIN: string = 'main';
const VIEW_CHAT: string = 'chat';
const VIEW_SETTINGS: string = 'settings';
const VIEW_API: string = 'api';

/**
 * Mirrors the Android app's navigation: a main screen with a history drawer and three bottom
 * tabs (My Models / Market / Benchmark); chat, settings and the API console are full-screen
 * views pushed on top of the main screen instead of permanent tabs.
 */
@Entry
@Component
struct Index {
  @State private currentView: string = VIEW_MAIN;
  @State private apiReturnView: string = VIEW_MAIN;
  @State private selectedTab: number = 0;
  @State private drawerShown: boolean = false;
  @State private requestedSessionId: string = '';
  @State private activeModelId: string = 'legacy-model';
  @State private activeModelName: string = '';
  @State private settingsRevision: number = 0;

  aboutToAppear(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(async () => {
      this.activeModelId = await SettingsStore.getInstance().getString(ACTIVE_MODEL_ID_KEY, 'legacy-model');
      this.settingsRevision = await SettingsStore.getInstance().getSettingsRevision();
      this.refreshModelName();
      const agreedVersion = await SettingsStore.getInstance().getNumber(PRIVACY_POLICY_KEY, 0);
      if (agreedVersion < PRIVACY_POLICY_VERSION) {
        setTimeout(() => this.showPrivacyPolicy(), 150);
      }
    });
  }

  onBackPress(): boolean {
    if (this.currentView !== VIEW_MAIN) {
      this.showMain();
      return true;
    }
    if (this.drawerShown) {
      this.drawerShown = false;
      return true;
    }
    return false;
  }

  private showMain(): void {
    this.currentView = VIEW_MAIN;
    this.refreshModelName();
  }

  private refreshModelName(): void {
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const active = models.find((model: ModelRecord) => model.id === this.activeModelId);
      this.activeModelName = active ? active.name : '';
    }).catch(() => {
    });
  }

  private showPrivacyPolicy(): void {
    let controller: CustomDialogController | null = null;
    controller = new CustomDialogController({
      builder: PrivacyPolicySheet({
        onAgree: () => {
          SettingsStore.getInstance().putNumber(PRIVACY_POLICY_KEY, PRIVACY_POLICY_VERSION).catch(() => {
          });
        },
        onDecline: () => {
          const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
          context.terminateSelf().catch(() => {
          });
        }
      }),
      autoCancel: false,
      alignment: DialogAlignment.Center,
      customStyle: true
    });
    controller.open();
  }

  private openModelPicker(): void {
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const ready: ModelRecord[] = models.filter((model: ModelRecord) => model.state === ModelState.READY);
      if (ready.length === 0) {
        return;
      }
      let controller: CustomDialogController | null = null;
      controller = new CustomDialogController({
        builder: ModelPickerSheet({
          models: ready,
          activeModelId: this.activeModelId,
          onPick: (model: ModelRecord) => {
            SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, model.id).then(() => {
              this.activeModelId = model.id;
              this.refreshModelName();
            }).catch(() => {
            });
          }
        }),
        autoCancel: true,
        alignment: DialogAlignment.Bottom,
        customStyle: true
      });
      controller.open();
    }).catch(() => {
    });
  }

  private openChatWithModel(modelId: string): void {
    this.activeModelId = modelId;
    this.requestedSessionId = '';
    this.currentView = VIEW_CHAT;
  }

  private openChatSession(sessionId: string): void {
    this.requestedSessionId = sessionId;
    this.drawerShown = false;
    this.currentView = VIEW_CHAT;
  }

  @Builder
  private mainToolbar() {
    Row({ space: 10 }) {
      Button('☰')
        .width(40)
        .height(40)
        .fontSize(18)
        .padding(0)
        .fontColor($r('app.color.text_primary'))
        .backgroundColor($r('app.color.surface_muted'))
        .onClick(() => {
          this.drawerShown = !this.drawerShown;
        })
      Row({ space: 6 }) {
        Text(this.activeModelName || this.activeModelId || tr('选择模型'))
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.text_primary'))
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .layoutWeight(1)
        Text('▾')
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
      }
      .layoutWeight(1)
      .height(40)
      .padding({ left: 12, right: 12 })
      .borderRadius(10)
      .backgroundColor($r('app.color.surface_muted'))
      .onClick(() => this.openModelPicker())

      Button(tr('设置'))
        .height(40)
        .fontSize(13)
        .padding({ left: 12, right: 12 })
        .fontColor($r('app.color.brand'))
        .backgroundColor($r('app.color.brand_tint'))
        .onClick(() => {
          this.currentView = VIEW_SETTINGS;
        })
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 10 })
    .backgroundColor($r('app.color.surface'))
  }

  @Builder
  private mainView() {
    Stack({ alignContent: Alignment.Start }) {
      Column() {
        this.mainToolbar()
        Divider()
          .color($r('app.color.divider'))
        Tabs({ barPosition: BarPosition.End, index: this.selectedTab }) {
          TabContent() {
            ModelsPage({
              activeModelId: this.activeModelId,
              onOpenChat: (modelId: string) => this.openChatWithModel(modelId),
              onOpenSettings: () => {
                this.currentView = VIEW_SETTINGS;
              }
            })
          }
          .tabBar(tr('模型'))

          TabContent() {
            MarketPage()
          }
          .tabBar(tr('市场'))

          TabContent() {
            BenchmarkPage({
              activeModelId: this.activeModelId,
              settingsRevision: this.settingsRevision
            })
          }
          .tabBar(tr('测试'))
        }
        .barHeight(54)
        .onChange((index: number) => {
          this.selectedTab = index;
        })
        .layoutWeight(1)
        .width('100%')
      }
      .width('100%')
      .height('100%')

      // History drawer, mirroring the Android DrawerLayout: scrim + left panel.
      if (this.drawerShown) {
        Column()
          .width('100%')
          .height('100%')
          .backgroundColor($r('app.color.overlay_close'))
          .onClick(() => {
            this.drawerShown = false;
          })

        Column() {
          HistoryPage({
            requestedSessionId: this.requestedSessionId,
            visible: this.drawerShown,
            onOpenChatSession: (sessionId: string) => this.openChatSession(sessionId)
          })
        }
        .width(320)
        .height('100%')
        .backgroundColor($r('app.color.background'))
        .transition(TransitionEffect.move(TransitionEdge.START).animation({ duration: 200 }))
      }
    }
    .width('100%')
    .height('100%')
  }

  build() {
    Column() {
      if (this.currentView === VIEW_CHAT) {
        ChatPage({
          requestedSessionId: this.requestedSessionId,
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => this.showMain(),
          onOpenBenchmark: () => {
            this.selectedTab = 2;
            this.showMain();
          },
          onOpenSettings: () => {
            this.currentView = VIEW_SETTINGS;
          },
          onOpenApi: () => {
            this.apiReturnView = VIEW_CHAT;
            this.currentView = VIEW_API;
          }
        })
      } else if (this.currentView === VIEW_SETTINGS) {
        SettingsPage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => this.showMain(),
          onOpenApi: () => {
            this.apiReturnView = VIEW_SETTINGS;
            this.currentView = VIEW_API;
          }
        })
      } else if (this.currentView === VIEW_API) {
        ApiConsolePage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => {
            this.currentView = this.apiReturnView;
          }
        })
      } else {
        this.mainView()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
  }
}
