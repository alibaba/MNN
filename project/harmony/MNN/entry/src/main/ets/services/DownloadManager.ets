import { common } from '@kit.AbilityKit';
import { request } from '@kit.BasicServicesKit';
import { fileIo, hash, statfs } from '@kit.CoreFileKit';
import { DownloadRepository } from '../data/DownloadRepository';
import { ModelRepository } from '../data/ModelRepository';
import {
  DownloadState,
  DownloadTaskRecord,
  MarketModelRecord,
  ModelDownloadStatus,
  ProviderFileRecord
} from '../domain/AppTypes';
import { ProviderAdapters } from './ProviderAdapters';

export type DownloadStatusListener = (status: ModelDownloadStatus) => void;

class ActiveDownload {
  modelId: string = '';
  modelName: string = '';
  provider: string = '';
  stagingRoot: string = '';
  files: ProviderFileRecord[] = [];
  tasks: request.agent.Task[] = [];
  records: DownloadTaskRecord[] = [];
  completedCount: number = 0;
  finishing: boolean = false;
  listener?: DownloadStatusListener;
  context?: common.UIAbilityContext;
  replace: boolean = false;
}

export class DownloadManager {
  private static instance: DownloadManager = new DownloadManager();
  private active: Map<string, ActiveDownload> = new Map<string, ActiveDownload>();
  private lastPersistAt: Map<string, number> = new Map<string, number>();

  static getInstance(): DownloadManager {
    return DownloadManager.instance;
  }

  isActive(modelId: string): boolean {
    return this.active.has(modelId);
  }

  async start(context: common.UIAbilityContext, model: MarketModelRecord, provider: string,
    listener: DownloadStatusListener, replace: boolean = false): Promise<void> {
    const repository = ProviderAdapters.repository(model, provider);
    if (!repository) {
      throw new Error('当前模型不支持所选来源');
    }
    const modelId = `${provider}/${repository}`;
    if (this.active.has(modelId)) {
      return;
    }
    if (replace) {
      // Drop stale completed records so start() does not take the resume path.
      await DownloadRepository.getInstance().removeModel(modelId);
    }

    const files = await ProviderAdapters.listFiles(model, provider);
    if (files.length === 0 || !files.some((file: ProviderFileRecord) => file.relativePath === 'config.json')) {
      throw new Error('仓库缺少可下载文件或 config.json');
    }
    let totalBytes = 0;
    files.forEach((file: ProviderFileRecord) => {
      totalBytes += file.sizeBytes;
    });
    const freeBytes = statfs.getFreeSizeSync(context.filesDir);
    if (totalBytes > 0 && freeBytes < totalBytes + 512 * 1024 * 1024) {
      throw new Error('可用空间不足，至少需要额外保留 512 MB');
    }

    const safeName = this.safeName(model.modelName);
    const stagingRoot = `${context.filesDir}/.mnn-downloads/${safeName}`;
    const downloadsRoot = `${context.filesDir}/.mnn-downloads`;
    if (!fileIo.accessSync(downloadsRoot)) {
      fileIo.mkdirSync(downloadsRoot, true);
    }
    const previous = await DownloadRepository.getInstance().list(modelId);
    if (previous.length > 0) {
      await this.resume(context, model, provider, listener);
      return;
    }
    if (fileIo.accessSync(stagingRoot)) {
      fileIo.rmdirSync(stagingRoot);
    }
    fileIo.mkdirSync(stagingRoot, true);

    const active = new ActiveDownload();
    active.modelId = modelId;
    active.modelName = model.modelName;
    active.provider = provider;
    active.stagingRoot = stagingRoot;
    active.files = files;
    active.listener = listener;
    active.context = context;
    active.replace = replace;
    this.active.set(modelId, active);

    try {
      for (let index = 0; index < files.length; index++) {
        const file = files[index];
        const targetPath = `${stagingRoot}/${file.relativePath}`;
        this.ensureParent(targetPath);
        const config: request.agent.Config = {
          action: request.agent.Action.DOWNLOAD,
          mode: request.agent.Mode.FOREGROUND,
          url: file.downloadUrl,
          title: model.modelName,
          description: file.relativePath,
          method: 'GET',
          headers: { 'Accept-Encoding': 'identity' },
          saveas: targetPath,
          overwrite: false,
          network: request.agent.Network.ANY,
          metered: false,
          roaming: false,
          retry: true,
          redirect: true,
          gauge: true,
          precise: true
        };
        const task = await request.agent.create(context, config);
        const record = new DownloadTaskRecord();
        record.id = task.tid;
        record.modelId = modelId;
        record.provider = provider;
        record.sourceUrl = file.downloadUrl;
        record.targetPath = targetPath;
        record.state = DownloadState.PENDING;
        record.totalBytes = file.sizeBytes;
        record.updatedAt = Date.now();
        await DownloadRepository.getInstance().save(record);
        active.tasks.push(task);
        active.records.push(record);
        this.attachTask(active, task, record);
      }
      this.notify(active, DownloadState.DOWNLOADING);
      for (let index = 0; index < active.tasks.length; index++) {
        await active.tasks[index].start();
      }
    } catch (error) {
      this.notify(active, DownloadState.FAILED, (error as Error).message);
      throw error as Error;
    }
  }

  /**
   * Pauses every in-flight download. Mirroring the Android client, this runs when a chat model is
   * loaded so download traffic does not compete with inference for bandwidth and CPU.
   */
  async pauseAll(): Promise<void> {
    const modelIds: string[] = [];
    this.active.forEach((value: ActiveDownload, key: string) => {
      modelIds.push(key);
    });
    for (let index = 0; index < modelIds.length; index++) {
      try {
        await this.pause(modelIds[index]);
      } catch (error) {
        // A failing task must not block loading the model.
      }
    }
  }

  async pause(modelId: string): Promise<void> {
    const active = this.active.get(modelId);
    if (!active) {
      return;
    }
    for (let index = 0; index < active.tasks.length; index++) {
      const task = active.tasks[index];
      const record = active.records.find((item: DownloadTaskRecord) => item.id === task.tid);
      if (record?.state !== DownloadState.COMPLETED) {
        await task.pause();
      }
    }
    await this.updateAll(active, DownloadState.PAUSED);
  }

  async resume(context: common.UIAbilityContext, model: MarketModelRecord, provider: string,
    listener: DownloadStatusListener): Promise<void> {
    const repository = ProviderAdapters.repository(model, provider);
    const modelId = `${provider}/${repository}`;
    let active = this.active.get(modelId);
    if (!active) {
      const records = await DownloadRepository.getInstance().list(modelId);
      if (records.length === 0) {
        await this.start(context, model, provider, listener);
        return;
      }
      active = new ActiveDownload();
      active.modelId = modelId;
      active.modelName = model.modelName;
      active.provider = provider;
      active.stagingRoot = `${context.filesDir}/.mnn-downloads/${this.safeName(model.modelName)}`;
      active.files = await ProviderAdapters.listFiles(model, provider);
      if (active.files.length !== records.length) {
        throw new Error('仓库文件列表已变化，请取消后重新下载');
      }
      active.listener = listener;
      active.context = context;
      for (let index = 0; index < records.length; index++) {
        const record = records[index];
        if (record.state === DownloadState.COMPLETED) {
          active.completedCount++;
          active.records.push(record);
          continue;
        }
        const task = await request.agent.getTask(context, record.id);
        active.tasks.push(task);
        active.records.push(record);
        this.attachTask(active, task, record);
      }
      this.active.set(modelId, active);
    }
    this.notify(active, DownloadState.DOWNLOADING);
    for (let index = 0; index < active.tasks.length; index++) {
      const task = active.tasks[index];
      const info = await request.agent.show(task.tid);
      if (info.progress.state === request.agent.State.PAUSED) {
        await task.resume();
      } else if (info.progress.state === request.agent.State.INITIALIZED ||
        info.progress.state === request.agent.State.STOPPED) {
        await task.start();
      } else if (info.progress.state === request.agent.State.COMPLETED) {
        const record = active.records.find((item: DownloadTaskRecord) => item.id === task.tid);
        if (record && record.state !== DownloadState.COMPLETED) {
          record.state = DownloadState.COMPLETED;
          record.downloadedBytes = record.totalBytes;
          record.updatedAt = Date.now();
          await DownloadRepository.getInstance().save(record);
          active.completedCount++;
        }
      } else if (info.progress.state === request.agent.State.FAILED ||
        info.progress.state === request.agent.State.REMOVED) {
        throw new Error('系统下载任务已失效，请取消后重试');
      }
    }
    if (active.completedCount === active.records.length && !active.finishing) {
      this.finish(active);
    }
  }

  async cancel(context: common.UIAbilityContext, modelId: string): Promise<void> {
    const active = this.active.get(modelId);
    const records = active ? active.records : await DownloadRepository.getInstance().list(modelId);
    for (let index = 0; index < records.length; index++) {
      try {
        await request.agent.remove(records[index].id);
      } catch (error) {
      }
    }
    const downloadsRoot = `${context.filesDir}/.mnn-downloads/`;
    let stagingRoot = active?.stagingRoot || '';
    if (!stagingRoot && records.length > 0 && records[0].targetPath.startsWith(downloadsRoot)) {
      const relativePath = records[0].targetPath.substring(downloadsRoot.length);
      const directoryName = relativePath.substring(0, relativePath.indexOf('/'));
      if (directoryName && directoryName.indexOf('..') < 0) {
        stagingRoot = `${downloadsRoot}${directoryName}`;
      }
    }
    if (stagingRoot.startsWith(downloadsRoot) && fileIo.accessSync(stagingRoot)) {
      fileIo.rmdirSync(stagingRoot);
    }
    await DownloadRepository.getInstance().removeModel(modelId);
    this.active.delete(modelId);
    if (active) {
      this.notify(active, DownloadState.PENDING);
    }
  }

  async status(modelId: string): Promise<ModelDownloadStatus> {
    const records = await DownloadRepository.getInstance().list(modelId);
    const status = new ModelDownloadStatus();
    status.modelId = modelId;
    if (records.length === 0) {
      return status;
    }
    status.state = DownloadState.COMPLETED;
    records.forEach((record: DownloadTaskRecord) => {
      status.downloadedBytes += record.downloadedBytes;
      status.totalBytes += record.totalBytes;
      if (record.state === DownloadState.FAILED) {
        status.state = DownloadState.FAILED;
      } else if (record.state === DownloadState.DOWNLOADING && status.state !== DownloadState.FAILED) {
        status.state = DownloadState.DOWNLOADING;
      } else if (record.state === DownloadState.PAUSED && status.state === DownloadState.COMPLETED) {
        status.state = DownloadState.PAUSED;
      } else if (record.state === DownloadState.PENDING && status.state === DownloadState.COMPLETED) {
        status.state = DownloadState.PENDING;
      }
    });
    return status;
  }

  private attachTask(active: ActiveDownload, task: request.agent.Task, record: DownloadTaskRecord): void {
    task.on('progress', (progress: request.agent.Progress) => {
      record.state = DownloadState.DOWNLOADING;
      record.downloadedBytes = progress.processed;
      if (progress.sizes.length > 0 && progress.sizes[0] > 0) {
        record.totalBytes = progress.sizes[0];
      }
      record.updatedAt = Date.now();
      const lastPersistAt = this.lastPersistAt.get(record.id) || 0;
      if (record.updatedAt - lastPersistAt >= 1000 || record.downloadedBytes === record.totalBytes) {
        this.lastPersistAt.set(record.id, record.updatedAt);
        DownloadRepository.getInstance().save(record);
      }
      this.notify(active, DownloadState.DOWNLOADING);
    });
    task.on('completed', (progress: request.agent.Progress) => {
      record.state = DownloadState.COMPLETED;
      record.downloadedBytes = record.totalBytes > 0 ? record.totalBytes : progress.processed;
      record.updatedAt = Date.now();
      DownloadRepository.getInstance().save(record).then(() => {
        active.completedCount++;
        if (active.completedCount === active.records.length) {
          this.finish(active);
        } else {
          this.notify(active, DownloadState.DOWNLOADING);
        }
      });
    });
    task.on('failed', (progress: request.agent.Progress) => {
      record.state = DownloadState.FAILED;
      record.downloadedBytes = progress.processed;
      record.updatedAt = Date.now();
      DownloadRepository.getInstance().save(record);
      this.notify(active, DownloadState.FAILED, '文件下载失败');
    });
  }

  private async updateAll(active: ActiveDownload, state: DownloadState): Promise<void> {
    for (let index = 0; index < active.records.length; index++) {
      active.records[index].state = state;
      active.records[index].updatedAt = Date.now();
      await DownloadRepository.getInstance().save(active.records[index]);
    }
    this.notify(active, state);
  }

  private async finish(active: ActiveDownload): Promise<void> {
    if (active.finishing) {
      return;
    }
    active.finishing = true;
    try {
      for (let index = 0; index < active.files.length; index++) {
        const file = active.files[index];
        const path = `${active.stagingRoot}/${file.relativePath}`;
        if (!fileIo.accessSync(path) || !fileIo.lstatSync(path).isFile()) {
          throw new Error(`下载文件缺失：${file.relativePath}`);
        }
        if (file.sizeBytes > 0 && fileIo.lstatSync(path).size !== file.sizeBytes) {
          throw new Error(`下载文件大小不匹配：${file.relativePath}`);
        }
        if (file.sha256) {
          const digest = await hash.hash(path, 'sha256');
          if (digest.toLowerCase() !== file.sha256.toLowerCase()) {
            throw new Error(`下载文件校验失败：${file.relativePath}`);
          }
        }
      }
      if (!active.context) {
        throw new Error('下载上下文不可用');
      }
      if (active.replace) {
        await ModelRepository.getInstance().replaceDownloadedModel(
          active.stagingRoot, active.context.filesDir, active.modelName
        );
      } else {
        await ModelRepository.getInstance().installDownloadedModel(
          active.stagingRoot, active.context.filesDir, active.modelName
        );
      }
      await this.updateAll(active, DownloadState.COMPLETED);
      this.active.delete(active.modelId);
    } catch (error) {
      await this.updateAll(active, DownloadState.FAILED);
      this.notify(active, DownloadState.FAILED, (error as Error).message);
    }
  }

  private notify(active: ActiveDownload, state: DownloadState, error: string = ''): void {
    if (!active.listener) {
      return;
    }
    const status = new ModelDownloadStatus();
    status.modelId = active.modelId;
    status.state = state;
    status.error = error;
    active.records.forEach((record: DownloadTaskRecord) => {
      status.downloadedBytes += record.downloadedBytes;
      status.totalBytes += record.totalBytes;
    });
    active.listener(status);
  }

  private ensureParent(path: string): void {
    const separator = path.lastIndexOf('/');
    if (separator <= 0) {
      return;
    }
    const parent = path.substring(0, separator);
    if (!fileIo.accessSync(parent)) {
      fileIo.mkdirSync(parent, true);
    }
  }

  private safeName(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, '-').replace(/^-+|-+$/g, '').substring(0, 64);
  }
}
