import { http } from '@kit.NetworkKit';
import { MarketModelRecord, ProviderFileRecord } from '../domain/AppTypes';

export const PROVIDER_HUGGING_FACE: string = 'HuggingFace';
export const PROVIDER_MODEL_SCOPE: string = 'ModelScope';
export const PROVIDER_MODELERS: string = 'Modelers';

interface HfLfsDocument {
  oid?: string;
  size?: number;
}

interface HfTreeDocument {
  type: string;
  path: string;
  size?: number;
  oid?: string;
  lfs?: HfLfsDocument;
}

interface ModelScopeFileDocument {
  Path: string;
  Size: number;
  Sha256?: string;
  Type: string;
}

interface ModelScopeDataDocument {
  Files?: ModelScopeFileDocument[];
}

interface ModelScopeResponseDocument {
  Success: boolean;
  Message?: string;
  Data?: ModelScopeDataDocument;
}

interface ModelersFileDocument {
  type: string;
  path: string;
  size: number;
  etag?: string;
}

interface ModelersDataDocument {
  tree: ModelersFileDocument[];
}

interface ModelersResponseDocument {
  code: string | number;
  msg?: string;
  data: ModelersDataDocument;
}

export class ProviderAdapters {
  static providers(model: MarketModelRecord): string[] {
    const providers: string[] = [];
    if (model.sources.ModelScope) {
      providers.push(PROVIDER_MODEL_SCOPE);
    }
    if (model.sources.HuggingFace) {
      providers.push(PROVIDER_HUGGING_FACE);
    }
    if (model.sources.Modelers) {
      providers.push(PROVIDER_MODELERS);
    }
    return providers;
  }

  static repository(model: MarketModelRecord, provider: string): string {
    if (provider === PROVIDER_MODEL_SCOPE) {
      return model.sources.ModelScope || '';
    }
    if (provider === PROVIDER_HUGGING_FACE) {
      return model.sources.HuggingFace || '';
    }
    if (provider === PROVIDER_MODELERS) {
      return model.sources.Modelers || '';
    }
    return '';
  }

  static webPage(model: MarketModelRecord, provider: string): string {
    const repository = ProviderAdapters.repository(model, provider);
    if (!repository) {
      return '';
    }
    if (provider === PROVIDER_HUGGING_FACE) {
      return `https://huggingface.co/${repository}`;
    }
    if (provider === PROVIDER_MODEL_SCOPE) {
      return `https://modelscope.cn/models/${repository}`;
    }
    if (provider === PROVIDER_MODELERS) {
      return `https://modelers.cn/models/${repository}`;
    }
    return '';
  }

  static async listFiles(model: MarketModelRecord, provider: string,
    token: string = ''): Promise<ProviderFileRecord[]> {
    const repository = ProviderAdapters.repository(model, provider);
    const parts = repository.split('/');
    if (parts.length !== 2 || !parts[0] || !parts[1]) {
      throw new Error('模型仓库地址格式无效');
    }
    if (provider === PROVIDER_MODEL_SCOPE) {
      return ProviderAdapters.listModelScopeFiles(repository, token);
    }
    if (provider === PROVIDER_HUGGING_FACE) {
      return ProviderAdapters.listHuggingFaceFiles(repository, token);
    }
    if (provider === PROVIDER_MODELERS) {
      return ProviderAdapters.listModelersFiles(parts[0], parts[1], '', token);
    }
    throw new Error('不支持的模型来源');
  }

  /** Headers used both by listing requests and by the download agent. */
  static authHeaders(provider: string, token: string): Record<string, string> {
    const headers: Record<string, string> = {
      'Accept': 'application/json',
      'Accept-Encoding': 'identity'
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      if (provider === PROVIDER_MODEL_SCOPE) {
        // ModelScope selects the credential via this header in addition to Bearer.
        headers['X-ModelScope-Token'] = token;
      }
    }
    return headers;
  }

  private static async listHuggingFaceFiles(repository: string, token: string): Promise<ProviderFileRecord[]> {
    const encodedRepository = ProviderAdapters.encodePath(repository);
    const documents = await ProviderAdapters.requestJson(
      `https://huggingface.co/api/models/${encodedRepository}/tree/main?recursive=true`,
      PROVIDER_HUGGING_FACE, token
    ) as HfTreeDocument[];
    const files: ProviderFileRecord[] = [];
    documents.forEach((document: HfTreeDocument) => {
      if (document.type !== 'file' || !ProviderAdapters.isSafePath(document.path)) {
        return;
      }
      const file = new ProviderFileRecord();
      file.relativePath = document.path;
      file.downloadUrl = `https://huggingface.co/${encodedRepository}/resolve/main/` +
        ProviderAdapters.encodePath(document.path);
      file.sizeBytes = document.lfs?.size || document.size || 0;
      const digest = document.lfs?.oid || '';
      file.sha256 = digest.startsWith('sha256:') ? digest.substring(7) :
        (digest.length === 64 ? digest : '');
      files.push(file);
    });
    return files;
  }

  private static async listModelScopeFiles(repository: string, token: string): Promise<ProviderFileRecord[]> {
    const encodedRepository = ProviderAdapters.encodePath(repository);
    const response = await ProviderAdapters.requestJson(
      `https://modelscope.cn/api/v1/models/${encodedRepository}/repo/files?Recursive=1`,
      PROVIDER_MODEL_SCOPE, token
    ) as ModelScopeResponseDocument;
    if (!response.Success || !response.Data?.Files) {
      throw new Error(response.Message || 'ModelScope 文件列表请求失败');
    }
    const files: ProviderFileRecord[] = [];
    response.Data.Files.forEach((document: ModelScopeFileDocument) => {
      if (document.Type === 'tree' || !ProviderAdapters.isSafePath(document.Path)) {
        return;
      }
      const file = new ProviderFileRecord();
      file.relativePath = document.Path;
      file.downloadUrl = `https://modelscope.cn/api/v1/models/${encodedRepository}/repo?FilePath=` +
        encodeURIComponent(document.Path);
      file.sizeBytes = document.Size || 0;
      file.sha256 = document.Sha256 || '';
      files.push(file);
    });
    return files;
  }

  private static async listModelersFiles(owner: string, repositoryName: string,
    directory: string, token: string): Promise<ProviderFileRecord[]> {
    const response = await ProviderAdapters.requestJson(
      `https://modelers.cn/api/v1/file/${encodeURIComponent(owner)}/${encodeURIComponent(repositoryName)}?path=` +
        encodeURIComponent(directory), PROVIDER_MODELERS, token
    ) as ModelersResponseDocument;
    if (!response.data?.tree) {
      throw new Error(response.msg || 'Modelers 文件列表请求失败');
    }
    const files: ProviderFileRecord[] = [];
    for (let index = 0; index < response.data.tree.length; index++) {
      const document = response.data.tree[index];
      if (!ProviderAdapters.isSafePath(document.path)) {
        continue;
      }
      if (document.type === 'dir') {
        const nested = await ProviderAdapters.listModelersFiles(owner, repositoryName, document.path, token);
        files.push(...nested);
        continue;
      }
      const file = new ProviderFileRecord();
      file.relativePath = document.path;
      file.downloadUrl = `https://modelers.cn/coderepo/web/v1/file/${encodeURIComponent(owner)}/` +
        `${encodeURIComponent(repositoryName)}/main/media/${ProviderAdapters.encodePath(document.path)}`;
      file.sizeBytes = document.size || 0;
      file.sha256 = document.etag && document.etag.length === 64 ? document.etag : '';
      files.push(file);
    }
    return files;
  }

  private static async requestJson(url: string, provider: string, token: string): Promise<Object> {
    const request = http.createHttp();
    try {
      const response = await request.request(url, {
        method: http.RequestMethod.GET,
        header: ProviderAdapters.authHeaders(provider, token),
        connectTimeout: 30000,
        readTimeout: 30000
      });
      if (response.responseCode === 401 || response.responseCode === 403) {
        throw new Error('仓库拒绝访问，请在设置中填写有效的私有仓库令牌');
      }
      if (response.responseCode < 200 || response.responseCode >= 300) {
        throw new Error(`仓库请求失败：HTTP ${response.responseCode}`);
      }
      if (typeof response.result === 'string') {
        return JSON.parse(response.result) as Object;
      }
      return response.result as Object;
    } finally {
      request.destroy();
    }
  }

  private static encodePath(path: string): string {
    return path.split('/').map((segment: string) => encodeURIComponent(segment)).join('/');
  }

  private static isSafePath(path: string): boolean {
    return path.length > 0 && !path.startsWith('/') && !path.startsWith('\\') && path.indexOf('..') < 0 &&
      path.indexOf('://') < 0;
  }
}
