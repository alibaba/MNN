import { common } from '@kit.AbilityKit';
import { util } from '@kit.ArkTS';
import { http } from '@kit.NetworkKit';
import fs from '@ohos.file.fs';
import { MarketCatalogRecord, MarketModelRecord, MarketSources } from '../domain/AppTypes';

const MARKET_ASSET: string = 'model_market.json';
const MARKET_CACHE: string = 'model_market_cache.json';
const MARKET_URL: string = 'https://meta.alicdn.com/data/mnn/apis/model_market.json';

interface MarketModelDocument {
  modelName: string;
  vendor: string;
  size_gb?: number;
  sizeB?: number;
  file_size?: number;
  fileSize?: number;
  description?: string;
  tags: string[];
  categories: string[];
  sources: MarketSources;
}

interface MarketCatalogDocument {
  version: string;
  quickFilterTags: string[];
  vendorOrder?: string[];
  models: MarketModelDocument[];
}

export class ModelMarketService {
  private static instance: ModelMarketService = new ModelMarketService();

  static getInstance(): ModelMarketService {
    return ModelMarketService.instance;
  }

  async load(context: common.UIAbilityContext, refreshNetwork: boolean = false): Promise<MarketCatalogRecord> {
    const bundled = this.parseCatalog(this.decode(await context.resourceManager.getRawFileContent(MARKET_ASSET)));
    const cached = this.loadCache(context.filesDir);
    let selected = cached && this.compareVersions(cached.version, bundled.version) >= 0 ? cached : bundled;
    if (refreshNetwork) {
      const remote = await this.fetchRemote();
      if (this.compareVersions(remote.version, selected.version) >= 0) {
        selected = remote;
        this.writeCache(context.filesDir, remote);
      }
    }
    return selected;
  }

  private async fetchRemote(): Promise<MarketCatalogRecord> {
    const request = http.createHttp();
    try {
      const response = await request.request(MARKET_URL, {
        method: http.RequestMethod.GET,
        header: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Accept': 'application/json'
        },
        connectTimeout: 10000,
        readTimeout: 30000
      });
      if (response.responseCode < 200 || response.responseCode >= 300) {
        throw new Error(`市场请求失败：HTTP ${response.responseCode}`);
      }
      const body = typeof response.result === 'string' ? response.result : JSON.stringify(response.result);
      return this.parseCatalog(body);
    } finally {
      request.destroy();
    }
  }

  private loadCache(filesDir: string): MarketCatalogRecord | undefined {
    const path = `${filesDir}/${MARKET_CACHE}`;
    if (!fs.accessSync(path)) {
      return undefined;
    }
    try {
      return this.parseCatalog(fs.readTextSync(path));
    } catch (error) {
      return undefined;
    }
  }

  private writeCache(filesDir: string, catalog: MarketCatalogRecord): void {
    const path = `${filesDir}/${MARKET_CACHE}`;
    const file = fs.openSync(path, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
    try {
      fs.writeSync(file.fd, JSON.stringify(catalog));
    } finally {
      fs.closeSync(file);
    }
  }

  private parseCatalog(value: string): MarketCatalogRecord {
    const document = JSON.parse(value) as MarketCatalogDocument;
    if (!document.version || !document.models) {
      throw new Error('模型市场数据格式无效');
    }
    const catalog = new MarketCatalogRecord();
    catalog.version = document.version;
    catalog.quickFilterTags = document.quickFilterTags || [];
    catalog.vendorOrder = document.vendorOrder || [];
    document.models.forEach((item: MarketModelDocument) => {
      if (!item.modelName || !item.vendor || !item.sources) {
        return;
      }
      const model = new MarketModelRecord();
      model.modelName = item.modelName;
      model.vendor = item.vendor;
      model.sizeB = item.size_gb || item.sizeB || 0;
      model.fileSize = item.file_size || item.fileSize || 0;
      model.description = item.description || '';
      model.tags = item.tags || [];
      model.categories = item.categories || [];
      model.sources = item.sources;
      catalog.models.push(model);
    });
    return catalog;
  }

  private decode(value: Uint8Array): string {
    return util.TextDecoder.create('utf-8', { ignoreBOM: true }).decodeWithStream(value);
  }

  private compareVersions(first: string, second: string): number {
    const firstParts = first.split('.');
    const secondParts = second.split('.');
    const length = Math.max(firstParts.length, secondParts.length);
    for (let index = 0; index < length; index++) {
      const firstValue = index < firstParts.length ? Number(firstParts[index]) : 0;
      const secondValue = index < secondParts.length ? Number(secondParts[index]) : 0;
      if (firstValue !== secondValue) {
        return firstValue > secondValue ? 1 : -1;
      }
    }
    return 0;
  }
}
