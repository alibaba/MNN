import { common, wantAgent } from '@kit.AbilityKit';
import { backgroundTaskManager } from '@kit.BackgroundTasksKit';

/**
 * Keeps the process eligible to run while the app is in the background.
 * Requires ohos.permission.KEEP_BACKGROUND_RUNNING; if the system refuses the request the
 * foreground feature still works, but may be suspended when the app leaves the foreground.
 *
 * Multiple features need this at the same time (model downloads and the local API server), so
 * ownership is reference counted: the continuous task starts on the first acquire and stops only
 * after the last owner releases it.
 */
export class BackgroundTaskKeeper {
  static readonly OWNER_DOWNLOAD: string = 'download';
  static readonly OWNER_API_SERVER: string = 'api';

  private static instance: BackgroundTaskKeeper = new BackgroundTaskKeeper();
  private context?: common.UIAbilityContext;
  private running: boolean = false;
  private starting?: Promise<boolean>;
  private owners: Set<string> = new Set<string>();

  static getInstance(): BackgroundTaskKeeper {
    return BackgroundTaskKeeper.instance;
  }

  async acquire(context: common.UIAbilityContext, owner: string): Promise<boolean> {
    this.owners.add(owner);
    this.context = context;
    if (this.running) {
      return true;
    }
    if (!this.starting) {
      const pending = this.startTask(context);
      this.starting = pending;
      pending.then(() => {
        if (this.starting === pending) {
          this.starting = undefined;
        }
      }, () => {
        if (this.starting === pending) {
          this.starting = undefined;
        }
      });
    }
    return this.starting;
  }

  async release(owner: string): Promise<void> {
    this.owners.delete(owner);
    if (this.owners.size > 0) {
      return;
    }
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

  isRunning(): boolean {
    return this.running;
  }

  isHeldBy(owner: string): boolean {
    return this.owners.has(owner);
  }

  private async startTask(context: common.UIAbilityContext): Promise<boolean> {
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
      // Continuous tasks may be unavailable (permission, device policy); keep the feature running.
      console.error(`[BackgroundTaskKeeper] startBackgroundRunning failed: ${JSON.stringify(error)}`);
      this.running = false;
      return false;
    }
  }
}