import { common } from '@kit.AbilityKit';
import { AppDatabase } from '../data/AppDatabase';
import { ChatRepository } from '../data/ChatRepository';
import { ModelRepository } from '../data/ModelRepository';
import { ModelState } from '../domain/AppTypes';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from './SettingsStore';

export class AppInitializer {
  private static initialization?: Promise<void>;

  static initialize(context: common.UIAbilityContext): Promise<void> {
    if (!AppInitializer.initialization) {
      AppInitializer.initialization = AppInitializer.initializeInternal(context);
    }
    return AppInitializer.initialization;
  }

  private static async initializeInternal(context: common.UIAbilityContext): Promise<void> {
    await AppDatabase.getInstance().initialize(context);
    await SettingsStore.getInstance().initialize(context);
    await ChatRepository.getInstance().recoverInterruptedGenerations();
    const legacyModel = await ModelRepository.getInstance().registerLegacyModel(context.filesDir);
    const activeModelId = await SettingsStore.getInstance().getString(ACTIVE_MODEL_ID_KEY);
    const activeModel = activeModelId ? await ModelRepository.getInstance().getModel(activeModelId) : undefined;
    if (!activeModel || activeModel.state !== ModelState.READY) {
      await SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, legacyModel.id);
    }
  }
}
