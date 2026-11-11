import { tr } from '../services/I18n';
import relationalStore from '@ohos.data.relationalStore';
import fs from '@ohos.file.fs';
import fileUri from '@ohos.file.fileuri';
import statvfs from '@ohos.file.statvfs';
import { AppDatabase } from './AppDatabase';
import { ModelCapabilities, ModelRecord, ModelState, ModelStorageStats } from '../domain/AppTypes';
import { SplitFileMerger } from '../services/SplitFileMerger';

const LEGACY_MODEL_ID: string = 'legacy-model';
const MANAGED_DIRECTORY: string = '.mnnmodels';
const TRASH_DIRECTORY: string = '.trash';

interface ModelConfigDocument {
  model_name?: string;
  base_dir?: string;
  llm_config?: string;
  llm_model?: string;
  llm_weight?: string;
  tokenizer_file?: string;
  embedding_file?: string;
  is_visual?: boolean;
  is_audio?: boolean;
  has_talker?: boolean;
  model_type?: string;
  tokenizer?: string;
  connector_model?: string;
  connector_weight?: string;
  projector_model?: string;
  projector_weight?: string;
  transformer_model?: string;
  transformer_weight?: string;
  vae_decoder_model?: string;
  vae_decoder_weight?: string;
  vae_encoder_model?: string;
  vae_encoder_weight?: string;
}

interface ModelValidationResult {
  valid: boolean;
  name: string;
  error: string;
}

export class ModelRepository {
  private static instance: ModelRepository = new ModelRepository();

  static getInstance(): ModelRepository {
    return ModelRepository.instance;
  }

  async registerLegacyModel(filesDir: string): Promise<ModelRecord> {
    const rootPath = `${filesDir}/model`;
    const configPath = `${rootPath}/config.json`;
    const previous = await this.getModel(LEGACY_MODEL_ID);
    const model = new ModelRecord();
    model.id = LEGACY_MODEL_ID;
    model.name = previous?.name || 'Qwen3-0.6B';
    model.configPath = configPath;
    model.rootPath = rootPath;
    const validation = this.validateModel(rootPath);
    model.state = validation.valid ? ModelState.READY : ModelState.INVALID;
    model.isLegacy = true;
    model.isProtected = true;
    model.isPinned = previous?.isPinned || false;
    model.lastUsedAt = previous?.lastUsedAt || 0;
    model.createdAt = previous?.createdAt || Date.now();
    model.sizeBytes = fs.accessSync(rootPath) ? this.calculateDirectorySize(rootPath) : 0;
    await this.upsert(model);
    return model;
  }

  async scanModels(filesDir: string): Promise<ModelRecord[]> {
    await this.registerLegacyModel(filesDir);
    const managedRoot = this.prepareManagedRoot(filesDir);
    this.cleanupTrash(managedRoot);

    const discoveredIds: string[] = [];
    const entries = fs.listFileSync(managedRoot);
    for (let index = 0; index < entries.length; index++) {
      const directoryName = entries[index];
      if (directoryName === TRASH_DIRECTORY) {
        continue;
      }
      const rootPath = `${managedRoot}/${directoryName}`;
      const stat = fs.lstatSync(rootPath);
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        continue;
      }
      const model = await this.createManagedRecord(rootPath, directoryName);
      discoveredIds.push(model.id);
      await this.upsert(model);
    }

    const storedModels = await this.listModels();
    for (let index = 0; index < storedModels.length; index++) {
      const model = storedModels[index];
      if (!model.isLegacy && model.rootPath.startsWith(`${managedRoot}/`) && discoveredIds.indexOf(model.id) < 0) {
        await this.deleteRecord(model.id);
      }
    }
    return this.listModels();
  }

  async importModel(sourceUri: string, filesDir: string, cacheDir: string): Promise<ModelRecord> {
    const managedRoot = this.prepareManagedRoot(filesDir);
    const stagingRoot = `${cacheDir}/mnn-import-${Date.now()}`;
    let targetRoot = '';
    let movedToTarget = false;
    fs.mkdirSync(stagingRoot, true);
    try {
      await fs.copy(sourceUri, fileUri.getUriFromPath(stagingRoot));
      const modelRoot = this.findModelRoot(stagingRoot);
      if (this.containsSymbolicLink(modelRoot)) {
        throw new Error(tr('模型目录不能包含符号链接'));
      }
      await SplitFileMerger.mergeIfNeeded(modelRoot);
      const directoryName = this.sanitizeName(this.baseName(modelRoot));
      if (!directoryName || directoryName === TRASH_DIRECTORY || directoryName.startsWith('.partial-') ||
        directoryName.indexOf('.deleting-') >= 0) {
        throw new Error(tr('无法从所选目录生成安全的模型名称'));
      }
      targetRoot = `${managedRoot}/${directoryName}`;
      if (fs.accessSync(targetRoot)) {
        throw new Error(`${tr('模型 ')}${directoryName}${tr(' 已存在')}`);
      }
      const validation = this.validateModel(modelRoot);
      if (!validation.valid) {
        throw new Error(validation.error);
      }
      fs.renameSync(modelRoot, targetRoot);
      movedToTarget = true;
      const model = await this.createManagedRecord(targetRoot, directoryName);
      await this.upsert(model);
      if (fs.accessSync(stagingRoot)) {
        fs.rmdirSync(stagingRoot);
      }
      return model;
    } catch (error) {
      if (movedToTarget && targetRoot && fs.accessSync(targetRoot)) {
        fs.rmdirSync(targetRoot);
      }
      if (fs.accessSync(stagingRoot)) {
        fs.rmdirSync(stagingRoot);
      }
      throw error as Error;
    }
  }

  async installDownloadedModel(stagingRoot: string, filesDir: string, name: string): Promise<ModelRecord> {
    const downloadsRoot = `${filesDir}/.mnn-downloads/`;
    if (!stagingRoot.startsWith(downloadsRoot) || this.containsSymbolicLink(stagingRoot)) {
      throw new Error(tr('下载暂存目录无效'));
    }
    const directoryName = this.sanitizeName(name);
    if (!directoryName || directoryName === TRASH_DIRECTORY || directoryName.startsWith('.partial-')) {
      throw new Error(tr('模型名称无效'));
    }
    const validation = this.validateModel(stagingRoot);
    if (!validation.valid) {
      throw new Error(validation.error);
    }
    const managedRoot = this.prepareManagedRoot(filesDir);
    const targetRoot = `${managedRoot}/${directoryName}`;
    if (fs.accessSync(targetRoot)) {
      throw new Error(`${tr('模型 ')}${directoryName}${tr(' 已存在')}`);
    }
    fs.renameSync(stagingRoot, targetRoot);
    try {
      const model = await this.createManagedRecord(targetRoot, directoryName);
      await this.upsert(model);
      return model;
    } catch (error) {
      if (fs.accessSync(targetRoot) && !fs.accessSync(stagingRoot)) {
        fs.renameSync(targetRoot, stagingRoot);
      }
      throw error as Error;
    }
  }

  /**
   * Installs a freshly downloaded copy over an existing model directory. The previous version is
   * moved into the trash first so a failed swap can be rolled back; trash is cleared on next scan.
   */
  async replaceDownloadedModel(stagingRoot: string, filesDir: string, name: string): Promise<ModelRecord> {
    const downloadsRoot = `${filesDir}/.mnn-downloads/`;
    if (!stagingRoot.startsWith(downloadsRoot) || this.containsSymbolicLink(stagingRoot)) {
      throw new Error(tr('下载暂存目录无效'));
    }
    const directoryName = this.sanitizeName(name);
    if (!directoryName || directoryName === TRASH_DIRECTORY || directoryName.startsWith('.partial-')) {
      throw new Error(tr('模型名称无效'));
    }
    const validation = this.validateModel(stagingRoot);
    if (!validation.valid) {
      throw new Error(validation.error);
    }
    const managedRoot = this.prepareManagedRoot(filesDir);
    const targetRoot = `${managedRoot}/${directoryName}`;
    const trashRoot = `${managedRoot}/${TRASH_DIRECTORY}`;
    if (!fs.accessSync(trashRoot)) {
      fs.mkdirSync(trashRoot, true);
    }
    const backupRoot = `${trashRoot}/${directoryName}-${Date.now()}`;
    let movedExisting = false;
    if (fs.accessSync(targetRoot)) {
      fs.renameSync(targetRoot, backupRoot);
      movedExisting = true;
    }
    try {
      fs.renameSync(stagingRoot, targetRoot);
      const model = await this.createManagedRecord(targetRoot, directoryName);
      await this.upsert(model);
      return model;
    } catch (error) {
      if (fs.accessSync(targetRoot)) {
        try {
          fs.renameSync(targetRoot, stagingRoot);
        } catch (rollbackError) {
        }
      }
      if (movedExisting && !fs.accessSync(targetRoot)) {
        try {
          fs.renameSync(backupRoot, targetRoot);
        } catch (rollbackError) {
        }
      }
      throw error as Error;
    }
  }

  async listModels(): Promise<ModelRecord[]> {
    const predicates = new relationalStore.RdbPredicates('models');
    predicates.orderByDesc('is_pinned').orderByDesc('last_used_at');
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    const models: ModelRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const model = this.readModel(result);
        model.sizeBytes = fs.accessSync(model.rootPath) ? this.calculateDirectorySize(model.rootPath) : 0;
        models.push(model);
      }
    } finally {
      result.close();
    }
    return models;
  }

  async getModel(modelId: string): Promise<ModelRecord | undefined> {
    const predicates = new relationalStore.RdbPredicates('models');
    predicates.equalTo('id', modelId);
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    try {
      if (!result.goToFirstRow()) {
        return undefined;
      }
      const model = this.readModel(result);
      model.sizeBytes = fs.accessSync(model.rootPath) ? this.calculateDirectorySize(model.rootPath) : 0;
      return model;
    } finally {
      result.close();
    }
  }

  async getCapabilities(modelId: string): Promise<ModelCapabilities> {
    const capabilities = new ModelCapabilities();
    const model = await this.getModel(modelId);
    if (!model || model.state !== ModelState.READY) {
      capabilities.attachmentReason = tr('模型不可用');
      return capabilities;
    }
    try {
      const config = JSON.parse(fs.readTextSync(model.configPath)) as ModelConfigDocument;
      if (config.model_type === 'diffusion') {
        capabilities.diffusion = true;
        // Editing-style diffusion models consume the picked image as the img2img source.
        capabilities.vision = true;
        capabilities.attachmentReason = '';
        return capabilities;
      }
      const llmConfigName = config.llm_config || 'llm_config.json';
      const llmConfigPath = `${model.rootPath}/${llmConfigName.startsWith('./') ? llmConfigName.substring(2) :
        llmConfigName}`;
      if (this.isSafeRelativePath(llmConfigName) && fs.accessSync(llmConfigPath)) {
        const llmConfig = JSON.parse(fs.readTextSync(llmConfigPath)) as ModelConfigDocument;
        config.is_visual = llmConfig.is_visual === undefined ? config.is_visual : llmConfig.is_visual;
        config.is_audio = llmConfig.is_audio === undefined ? config.is_audio : llmConfig.is_audio;
        config.has_talker = llmConfig.has_talker === undefined ? config.has_talker : llmConfig.has_talker;
      }
      capabilities.vision = config.is_visual === true;
      capabilities.audioInput = config.is_audio === true;
      capabilities.audioOutput = config.has_talker === true;
      // Video is sampled into still frames on the device, so any vision model can consume it.
      capabilities.videoInput = capabilities.vision;
      if (capabilities.videoInput) {
        capabilities.videoReason = '';
      }
      if (capabilities.vision || capabilities.audioInput) {
        capabilities.attachmentReason = '';
      }
      return capabilities;
    } catch (error) {
      capabilities.attachmentReason = tr('无法读取模型能力配置');
      return capabilities;
    }
  }

  async markUsed(modelId: string): Promise<void> {
    const values: relationalStore.ValuesBucket = { last_used_at: Date.now() };
    const predicates = new relationalStore.RdbPredicates('models');
    predicates.equalTo('id', modelId);
    await AppDatabase.getInstance().getStore().update(values, predicates);
  }

  async setPinned(modelId: string, pinned: boolean): Promise<void> {
    const values: relationalStore.ValuesBucket = { is_pinned: pinned ? 1 : 0 };
    const predicates = new relationalStore.RdbPredicates('models');
    predicates.equalTo('id', modelId);
    await AppDatabase.getInstance().getStore().update(values, predicates);
  }

  async deleteModel(modelId: string, filesDir: string): Promise<void> {
    const model = await this.getModel(modelId);
    if (!model) {
      return;
    }
    if (model.isProtected || model.isLegacy) {
      throw new Error(tr('受保护的 legacy 模型不能删除'));
    }
    const managedRoot = this.prepareManagedRoot(filesDir);
    const expectedRoot = `${managedRoot}/${this.baseName(model.rootPath)}`;
    if (model.rootPath !== expectedRoot ||
      (fs.accessSync(model.rootPath) && fs.lstatSync(model.rootPath).isSymbolicLink())) {
      throw new Error(tr('拒绝删除受管模型目录之外的路径'));
    }

    const trashRoot = `${managedRoot}/${TRASH_DIRECTORY}`;
    if (!fs.accessSync(trashRoot)) {
      fs.mkdirSync(trashRoot);
    }
    const trashPath = `${trashRoot}/${this.baseName(model.rootPath)}-${Date.now()}`;
    let movedToTrash = false;
    try {
      if (fs.accessSync(model.rootPath)) {
        fs.renameSync(model.rootPath, trashPath);
        movedToTrash = true;
      }
      await this.deleteRecord(modelId);
    } catch (error) {
      if (movedToTrash && fs.accessSync(trashPath) && !fs.accessSync(model.rootPath)) {
        fs.renameSync(trashPath, model.rootPath);
      }
      throw error as Error;
    }
    if (movedToTrash && fs.accessSync(trashPath)) {
      fs.rmdirSync(trashPath);
    }
  }

  getStorageStats(filesDir: string, models: ModelRecord[]): ModelStorageStats {
    const stats = new ModelStorageStats();
    models.forEach((model: ModelRecord) => {
      stats.modelBytes += model.sizeBytes;
    });
    stats.freeBytes = statvfs.getFreeSizeSync(filesDir);
    stats.totalBytes = statvfs.getTotalSizeSync(filesDir);
    return stats;
  }

  private async createManagedRecord(rootPath: string, directoryName: string): Promise<ModelRecord> {
    const validation = this.validateModel(rootPath);
    const id = this.managedModelId(directoryName);
    const previous = await this.getModel(id);
    const model = new ModelRecord();
    model.id = id;
    model.name = validation.name || directoryName;
    model.configPath = `${rootPath}/config.json`;
    model.rootPath = rootPath;
    model.state = validation.valid ? ModelState.READY : ModelState.INVALID;
    model.isPinned = previous?.isPinned || false;
    model.lastUsedAt = previous?.lastUsedAt || 0;
    model.createdAt = previous?.createdAt || Date.now();
    model.sizeBytes = this.calculateDirectorySize(rootPath);
    return model;
  }

  private validateModel(rootPath: string): ModelValidationResult {
    const configPath = `${rootPath}/config.json`;
    if (!fs.accessSync(configPath)) {
      return { valid: false, name: '', error: tr('所选目录缺少 config.json') };
    }
    if (this.containsSymbolicLink(rootPath)) {
      return { valid: false, name: '', error: tr('模型目录不能包含符号链接') };
    }
    const configStat = fs.lstatSync(configPath);
    if (!configStat.isFile() || configStat.size === 0) {
      return { valid: false, name: '', error: tr('config.json 不是有效文件') };
    }
    try {
      const config = JSON.parse(fs.readTextSync(configPath)) as ModelConfigDocument;
      const configuredBase = config.base_dir || '';
      if (configuredBase && configuredBase !== '.' && configuredBase !== './' && configuredBase !== rootPath &&
        configuredBase !== `${rootPath}/`) {
        return { valid: false, name: config.model_name || '', error: tr('base_dir 必须指向模型根目录') };
      }
      const modelBaseRoot = rootPath;

      const llmConfigName = config.llm_config || 'llm_config.json';
      if (!this.isSafeRelativePath(llmConfigName)) {
        return { valid: false, name: config.model_name || '', error: tr('llm_config 路径不安全') };
      }
      const llmConfigRelative = llmConfigName.startsWith('./') ? llmConfigName.substring(2) : llmConfigName;
      const llmConfigPath = `${modelBaseRoot}/${llmConfigRelative}`;
      if (fs.accessSync(llmConfigPath)) {
        const llmConfig = JSON.parse(fs.readTextSync(llmConfigPath)) as ModelConfigDocument;
        config.model_name = llmConfig.model_name || config.model_name;
        config.llm_model = llmConfig.llm_model || config.llm_model;
        config.llm_weight = llmConfig.llm_weight || config.llm_weight;
        config.tokenizer_file = llmConfig.tokenizer_file || config.tokenizer_file;
        config.embedding_file = llmConfig.embedding_file || config.embedding_file;
      }

      if (config.model_type === 'diffusion') {
        // Diffusion models use a different artifact layout (text encoder, connector, DiT, VAE)
        // and have no top-level tokenizer.mtok, so validate against their own file list.
        const diffusionReferences: string[] = [];
        const candidates: (string | undefined)[] = [
          config.llm_model, config.llm_weight, config.tokenizer,
          config.connector_model, config.connector_weight,
          config.projector_model, config.projector_weight,
          config.transformer_model, config.transformer_weight,
          config.vae_decoder_model, config.vae_decoder_weight,
          config.vae_encoder_model, config.vae_encoder_weight
        ];
        candidates.forEach((candidate: string | undefined) => {
          if (candidate) {
            diffusionReferences.push(candidate);
          }
        });
        if (diffusionReferences.length === 0) {
          return { valid: false, name: config.model_name || '', error: tr('扩散模型配置缺少模型文件列表') };
        }
        for (let index = 0; index < diffusionReferences.length; index++) {
          const reference = diffusionReferences[index];
          if (!this.isSafeRelativePath(reference)) {
            return { valid: false, name: config.model_name || '', error: `${tr('模型文件路径不安全：')}${reference}` };
          }
          const relativePath = reference.startsWith('./') ? reference.substring(2) : reference;
          const artifactPath = `${modelBaseRoot}/${relativePath}`;
          if (!fs.accessSync(artifactPath)) {
            if (SplitFileMerger.coversFile(modelBaseRoot, relativePath)) {
              continue;
            }
            return { valid: false, name: config.model_name || '', error: `${tr('模型文件缺失：')}${reference}` };
          }
          const artifactStat = fs.lstatSync(artifactPath);
          if (!artifactStat.isFile() || artifactStat.size === 0) {
            return { valid: false, name: config.model_name || '', error: `${tr('模型文件无效：')}${reference}` };
          }
        }
        return { valid: true, name: config.model_name || '', error: '' };
      }

      const references: string[] = [
        config.tokenizer_file || 'tokenizer.txt',
        config.llm_model || 'llm.mnn',
        config.llm_weight || 'llm.mnn.weight'
      ];
      if (config.embedding_file) {
        references.push(config.embedding_file);
      }
      for (let index = 0; index < references.length; index++) {
        const reference = references[index];
        if (!this.isSafeRelativePath(reference)) {
          return { valid: false, name: config.model_name || '', error: `${tr('模型文件路径不安全：')}${reference}` };
        }
        const relativePath = reference.startsWith('./') ? reference.substring(2) : reference;
        const artifactPath = `${modelBaseRoot}/${relativePath}`;
        if (!fs.accessSync(artifactPath)) {
          if (SplitFileMerger.coversFile(modelBaseRoot, relativePath)) {
            // Present as split chunks; merged lazily before loading.
            continue;
          }
          return { valid: false, name: config.model_name || '', error: `${tr('模型文件缺失：')}${reference}` };
        }
        const artifactStat = fs.lstatSync(artifactPath);
        if (!artifactStat.isFile() || artifactStat.size === 0) {
          return { valid: false, name: config.model_name || '', error: `${tr('模型文件无效：')}${reference}` };
        }
      }
      return { valid: true, name: config.model_name || '', error: '' };
    } catch (error) {
      return { valid: false, name: '', error: tr('config.json 或 llm_config.json 格式无效') };
    }
  }

  private findModelRoot(stagingRoot: string): string {
    if (fs.accessSync(`${stagingRoot}/config.json`)) {
      return stagingRoot;
    }
    const entries = fs.listFileSync(stagingRoot);
    for (let index = 0; index < entries.length; index++) {
      const candidate = `${stagingRoot}/${entries[index]}`;
      if (fs.statSync(candidate).isDirectory() && fs.accessSync(`${candidate}/config.json`)) {
        return candidate;
      }
    }
    throw new Error(tr('所选目录中未找到 config.json'));
  }

  private prepareManagedRoot(filesDir: string): string {
    const managedRoot = `${filesDir}/${MANAGED_DIRECTORY}`;
    if (!fs.accessSync(managedRoot)) {
      fs.mkdirSync(managedRoot, true);
    }
    const stat = fs.lstatSync(managedRoot);
    if (stat.isSymbolicLink() || !stat.isDirectory()) {
      throw new Error(tr('受管模型根目录无效'));
    }
    return managedRoot;
  }

  private cleanupTrash(managedRoot: string): void {
    const trashRoot = `${managedRoot}/${TRASH_DIRECTORY}`;
    if (!fs.accessSync(trashRoot)) {
      return;
    }
    const stat = fs.lstatSync(trashRoot);
    if (stat.isSymbolicLink() || !stat.isDirectory()) {
      throw new Error(tr('模型回收目录无效'));
    }
    const entries = fs.listFileSync(trashRoot);
    for (let index = 0; index < entries.length; index++) {
      const trashPath = `${trashRoot}/${entries[index]}`;
      if (fs.lstatSync(trashPath).isDirectory()) {
        fs.rmdirSync(trashPath);
      } else {
        fs.unlinkSync(trashPath);
      }
    }
  }

  private containsSymbolicLink(path: string): boolean {
    const stat = fs.lstatSync(path);
    if (stat.isSymbolicLink()) {
      return true;
    }
    if (!stat.isDirectory()) {
      return false;
    }
    const entries = fs.listFileSync(path);
    for (let index = 0; index < entries.length; index++) {
      if (this.containsSymbolicLink(`${path}/${entries[index]}`)) {
        return true;
      }
    }
    return false;
  }

  private calculateDirectorySize(path: string): number {
    const stat = fs.lstatSync(path);
    if (stat.isSymbolicLink()) {
      return 0;
    }
    if (!stat.isDirectory()) {
      return stat.size;
    }
    let size = 0;
    const entries = fs.listFileSync(path);
    for (let index = 0; index < entries.length; index++) {
      size += this.calculateDirectorySize(`${path}/${entries[index]}`);
    }
    return size;
  }

  private isSafeRelativePath(path: string): boolean {
    return path.length > 0 && !path.startsWith('/') && !path.startsWith('\\') && path.indexOf('..') < 0 &&
      path.indexOf('://') < 0;
  }

  private sanitizeName(name: string): string {
    return name.replace(/[^a-zA-Z0-9._-]/g, '-').replace(/^-+|-+$/g, '').substring(0, 64);
  }

  private managedModelId(directoryName: string): string {
    let hash = 0;
    for (let index = 0; index < directoryName.length; index++) {
      hash = (hash * 31 + directoryName.charCodeAt(index)) % 2147483647;
    }
    return `managed-${this.sanitizeName(directoryName).substring(0, 48)}-${hash.toString(16)}`;
  }

  private baseName(path: string): string {
    const normalized = path.endsWith('/') ? path.substring(0, path.length - 1) : path;
    return normalized.substring(normalized.lastIndexOf('/') + 1);
  }

  private async upsert(model: ModelRecord): Promise<void> {
    const existing = await this.getModel(model.id);
    const values = this.modelValues(model);
    if (existing) {
      const predicates = new relationalStore.RdbPredicates('models');
      predicates.equalTo('id', model.id);
      await AppDatabase.getInstance().getStore().update(values, predicates);
    } else {
      await AppDatabase.getInstance().getStore().insert('models', values);
    }
  }

  private async deleteRecord(modelId: string): Promise<void> {
    const predicates = new relationalStore.RdbPredicates('models');
    predicates.equalTo('id', modelId);
    await AppDatabase.getInstance().getStore().delete(predicates);
  }

  private readModel(result: relationalStore.ResultSet): ModelRecord {
    const model = new ModelRecord();
    model.id = result.getString(result.getColumnIndex('id'));
    model.name = result.getString(result.getColumnIndex('name'));
    model.configPath = result.getString(result.getColumnIndex('config_path'));
    model.rootPath = result.getString(result.getColumnIndex('root_path'));
    model.state = result.getString(result.getColumnIndex('state')) as ModelState;
    model.isLegacy = result.getLong(result.getColumnIndex('is_legacy')) === 1;
    model.isProtected = result.getLong(result.getColumnIndex('is_protected')) === 1;
    model.isPinned = result.getLong(result.getColumnIndex('is_pinned')) === 1;
    model.lastUsedAt = result.getLong(result.getColumnIndex('last_used_at'));
    model.createdAt = result.getLong(result.getColumnIndex('created_at'));
    return model;
  }

  private modelValues(model: ModelRecord): relationalStore.ValuesBucket {
    return {
      id: model.id,
      name: model.name,
      config_path: model.configPath,
      root_path: model.rootPath,
      state: model.state,
      is_legacy: model.isLegacy ? 1 : 0,
      is_protected: model.isProtected ? 1 : 0,
      is_pinned: model.isPinned ? 1 : 0,
      last_used_at: model.lastUsedAt,
      created_at: model.createdAt
    };
  }
}
