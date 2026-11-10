import { common, wantAgent } from '@kit.AbilityKit';
import { backgroundTaskManager } from '@kit.BackgroundTasksKit';

/**
 * Keeps the process eligible to keep transferring model files while the app is in the background.
 * Requires ohos.permission.KEEP_BACKGROUND_RUNNING; if the system refuses the request downloads
 * still work, but may be suspended when the app leaves the foreground.
 */
export class BackgroundTaskKeeper {
  private static instance: BackgroundTaskKeeper = new BackgroundTaskKeeper();
  private context?: common.UIAbilityContext;
  private running: boolean = false;

  static getInstance(): BackgroundTaskKeeper {
    return BackgroundTaskKeeper.instance;
  }

  async acquire(context: common.UIAbilityContext): Promise<boolean> {
    if (this.running) {
      return true;
    }
    this.context = context;
    try {
      const info: wantAgent.WantAgentInfo = {
        wants: [{
          bundleName: context.abilityInfo.bundleName,
          abilityName: context.abilityInfo.name
        }],
        actionType: wantAgent.OperationType.START_ABILITY,
        requestCode: 0,
        wantAgentFlags: [wantAgent.WantAgentFlags.UPDATE_PRESENT_FLAG]
      };
      const agent = await wantAgent.getWantAgent(info);
      await backgroundTaskManager.startBackgroundRunning(context,
        backgroundTaskManager.BackgroundMode.DATA_TRANSFER, agent);
      this.running = true;
      return true;
    } catch (error) {
      // Continuous tasks may be unavailable (permission, device policy); keep downloading.
      this.running = false;
      return false;
    }
  }

  async release(): Promise<void> {
    if (!this.running || !this.context) {
      return;
    }
    const context = this.context;
    this.running = false;
    try {
      await backgroundTaskManager.stopBackgroundRunning(context);
    } catch (error) {
    }
  }
}