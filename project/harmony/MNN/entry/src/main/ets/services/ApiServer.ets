import { common } from '@kit.AbilityKit';
import { util } from '@kit.ArkTS';
import { socket } from '@kit.NetworkKit';
import { ModelRepository } from '../data/ModelRepository';
import { ModelState } from '../domain/AppTypes';
import {
  GenerationCoordinator,
  NativeChatMessage,
  NativeGenerateRequest,
  NativeGenerateResult,
  NativeTarget
} from './GenerationCoordinator';
import { SettingsStore } from './SettingsStore';
import { BackgroundTaskKeeper } from './BackgroundTaskKeeper';

const API_TOKEN_KEY: string = 'local_api_token';
export const API_BIND_ALL_KEY: string = 'local_api_bind_all';
export const API_AUTH_ENABLED_KEY: string = 'local_api_auth_enabled';
export const API_CORS_ENABLED_KEY: string = 'local_api_cors_enabled';
export const API_CORS_ORIGINS_KEY: string = 'local_api_cors_origins';
export const API_HTTPS_URL_KEY: string = 'local_api_https_url';
export const API_TOKEN_SETTING_KEY: string = API_TOKEN_KEY;
const MAX_HEADER_BYTES: number = 16 * 1024;
const MAX_BODY_BYTES: number = 1024 * 1024;
const MAX_CONNECTIONS: number = 16;

interface ApiMessageDocument {
  role: string;
  content: string;
}

interface ChatRequestDocument {
  model?: string;
  messages: ApiMessageDocument[];
  system?: string;
  max_tokens?: number;
  stream?: boolean;
}

interface CancelRequestDocument {
  request_id: string;
}

class ParsedRequest {
  method: string = '';
  path: string = '';
  headers: Map<string, string> = new Map<string, string>();
  body: string = '';
}

export class ApiRequestLog {
  time: string = '';
  method: string = '';
  path: string = '';
  status: number = 0;
  durationMs: number = 0;
}

class ClientState {
  connection: socket.TCPSocketConnection;
  buffer: Uint8Array = new Uint8Array(0);
  handled: boolean = false;
  requestId: string = '';
  origin: string = '';
  lastMethod: string = '';
  lastPath: string = '';
  lastStartMs: number = 0;
  writeTail: Promise<void> = Promise.resolve();

  constructor(connection: socket.TCPSocketConnection) {
    this.connection = connection;
  }
}

export class ApiServerStatus {
  running: boolean = false;
  address: string = '127.0.0.1';
  port: number = 8080;
  token: string = '';
  connections: number = 0;
  queueSize: number = 0;
  backgroundActive: boolean = false;
  requestLogs: ApiRequestLog[] = [];
}

export class ApiServer {
  private static instance: ApiServer = new ApiServer();
  private server?: socket.TCPSocketServer;
  private clients: Map<number, ClientState> = new Map<number, ClientState>();
  private context?: common.UIAbilityContext;
  private target?: NativeTarget;
  private modelId: string = '';
  private systemPrompt: string = '';
  private maxTokens: number = 256;
  private token: string = '';
  private port: number = 8080;
  private bindAll: boolean = false;
  private authEnabled: boolean = true;
  private corsEnabled: boolean = false;
  private corsOrigins: string = '';
  private requestLogs: ApiRequestLog[] = [];

  static getInstance(): ApiServer {
    return ApiServer.instance;
  }

  async start(context: common.UIAbilityContext, modelId: string, port: number = 8080): Promise<ApiServerStatus> {
    if (this.server) {
      return this.getStatus();
    }
    if (port < 1024 || port > 65535) {
      throw new Error('端口必须在 1024 到 65535 之间');
    }
    const model = await ModelRepository.getInstance().getModel(modelId);
    if (!model || model.state !== ModelState.READY) {
      throw new Error('API 服务模型不可用');
    }
    const settings = await SettingsStore.getInstance().getModelRuntimeSettings(model.id);
    const safeModelId = model.id.replace(/[^a-zA-Z0-9._-]/g, '-');
    this.target = {
      modelId: model.id,
      configPath: model.configPath,
      runtimeConfigJson: SettingsStore.getInstance().createRuntimeConfigJson(
        settings, `${context.cacheDir}/llm-${safeModelId}`
      )
    };
    this.modelId = model.id;
    this.systemPrompt = settings.systemPrompt;
    this.maxTokens = settings.maxTokens;
    this.context = context;
    this.token = await this.ensureToken();
    this.port = port;
    const store = SettingsStore.getInstance();
    this.bindAll = await store.getBoolean(API_BIND_ALL_KEY, false);
    this.authEnabled = await store.getBoolean(API_AUTH_ENABLED_KEY, true);
    this.corsEnabled = await store.getBoolean(API_CORS_ENABLED_KEY, false);
    this.corsOrigins = await store.getString(API_CORS_ORIGINS_KEY, '');

    const listenAddress = this.bindAll ? '0.0.0.0' : '127.0.0.1';
    const server = socket.constructTCPSocketServerInstance();
    await server.listen({ address: listenAddress, family: 1, port: port });
    const localAddress = await server.getLocalAddress();
    if (localAddress.port !== port || (!this.bindAll && localAddress.address !== '127.0.0.1')) {
      await server.close();
      throw new Error(`无法绑定地址 ${listenAddress}:${port}`);
    }
    server.on('connect', (connection: socket.TCPSocketConnection) => this.accept(connection));
    server.on('error', () => {
    });
    this.server = server;
    // Keep serving requests after the app moves to the background, like the Android foreground
    // service. The continuous task also surfaces the required system notification.
    await BackgroundTaskKeeper.getInstance().acquire(context, BackgroundTaskKeeper.OWNER_API_SERVER);
    return this.getStatus();
  }

  async stop(): Promise<void> {
    const clients = Array.from(this.clients.values());
    this.clients.clear();
    clients.forEach((client: ClientState) => {
      if (client.requestId) {
        GenerationCoordinator.getInstance().cancel(client.requestId);
      }
      client.connection.close();
    });
    if (this.server) {
      const server = this.server;
      this.server = undefined;
      server.off('connect');
      server.off('error');
      await server.close();
    }
    await BackgroundTaskKeeper.getInstance().release(BackgroundTaskKeeper.OWNER_API_SERVER);
  }

  getStatus(): ApiServerStatus {
    const status = new ApiServerStatus();
    status.running = !!this.server;
    status.port = this.port;
    status.token = this.token;
    status.connections = this.clients.size;
    status.queueSize = GenerationCoordinator.getInstance().queueSize();
    status.backgroundActive = this.server !== undefined &&
      BackgroundTaskKeeper.getInstance().isRunning() &&
      BackgroundTaskKeeper.getInstance().isHeldBy(BackgroundTaskKeeper.OWNER_API_SERVER);
    status.requestLogs = this.requestLogs.slice();
    return status;
  }

  private accept(connection: socket.TCPSocketConnection): void {
    if (this.clients.size >= MAX_CONNECTIONS) {
      this.sendRaw(connection, this.httpResponse(503, 'Service Unavailable', '{"error":"too many connections"}'))
        .finally(() => connection.close());
      return;
    }
    const state = new ClientState(connection);
    this.clients.set(connection.clientId, state);
    connection.on('message', (message: socket.SocketMessageInfo) => {
      this.onMessage(state, new Uint8Array(message.message));
    });
    connection.on('close', () => this.closeClient(state));
    connection.on('error', () => this.closeClient(state));
  }

  private onMessage(client: ClientState, bytes: Uint8Array): void {
    if (client.handled) {
      return;
    }
    client.buffer = this.append(client.buffer, bytes);
    if (client.buffer.byteLength > MAX_HEADER_BYTES + MAX_BODY_BYTES) {
      client.handled = true;
      this.sendAndClose(client, this.httpResponse(413, 'Payload Too Large', '{"error":"request too large"}', client.origin));
      return;
    }
    try {
      const request = this.parseRequest(client.buffer);
      if (!request) {
        return;
      }
      client.handled = true;
      this.dispatch(client, request).catch((error: Error) => {
        this.sendAndClose(client, this.errorResponse(500, error.message, client.origin));
      });
    } catch (error) {
      client.handled = true;
      this.sendAndClose(client, this.errorResponse(400, (error as Error).message, client.origin));
    }
  }

  private parseRequest(buffer: Uint8Array): ParsedRequest | undefined {
    const headerEnd = this.findHeaderEnd(buffer);
    if (headerEnd < 0) {
      if (buffer.byteLength > MAX_HEADER_BYTES) {
        throw new Error('请求头过大');
      }
      return undefined;
    }
    const headerText = util.TextDecoder.create('utf-8', { fatal: true })
      .decodeToString(buffer.slice(0, headerEnd));
    const lines = headerText.split('\r\n');
    const requestLine = lines[0].split(' ');
    if (requestLine.length !== 3 || requestLine[2] !== 'HTTP/1.1') {
      throw new Error('无效 HTTP 请求行');
    }
    const request = new ParsedRequest();
    request.method = requestLine[0];
    request.path = requestLine[1];
    let contentLength = 0;
    for (let index = 1; index < lines.length; index++) {
      const separator = lines[index].indexOf(':');
      if (separator <= 0) {
        throw new Error('无效 HTTP 请求头');
      }
      const name = lines[index].substring(0, separator).trim().toLowerCase();
      const value = lines[index].substring(separator + 1).trim();
      if (request.headers.has(name)) {
        throw new Error(`重复请求头：${name}`);
      }
      request.headers.set(name, value);
    }
    if (!request.headers.has('host')) {
      throw new Error('缺少 Host 请求头');
    }
    if (request.headers.has('transfer-encoding')) {
      throw new Error('不支持 Transfer-Encoding 请求体');
    }
    if (request.headers.has('content-length')) {
      const value = request.headers.get('content-length') || '';
      if (!/^\d+$/.test(value)) {
        throw new Error('无效 Content-Length');
      }
      contentLength = Number(value);
      if (contentLength > MAX_BODY_BYTES) {
        throw new Error('请求体过大');
      }
    }
    const bodyOffset = headerEnd + 4;
    if (buffer.byteLength < bodyOffset + contentLength) {
      return undefined;
    }
    if (buffer.byteLength !== bodyOffset + contentLength) {
      throw new Error('不支持 HTTP 管线请求');
    }
    if (contentLength > 0) {
      request.body = util.TextDecoder.create('utf-8', { fatal: true })
        .decodeToString(buffer.slice(bodyOffset));
    }
    return request;
  }

  private async dispatch(client: ClientState, request: ParsedRequest): Promise<void> {
    client.origin = request.headers.get('origin') || '';
    client.lastMethod = request.method;
    client.lastPath = request.path;
    client.lastStartMs = Date.now();
    if (request.method === 'OPTIONS') {
      await this.sendAndClose(client, this.httpResponse(204, 'No Content', '', client.origin));
      return;
    }
    if (!this.authorized(request)) {
      await this.sendAndClose(client,
        this.httpResponse(401, 'Unauthorized', '{"error":"unauthorized"}', client.origin, 'WWW-Authenticate: Bearer\r\n'));
      return;
    }
    if (request.method === 'GET' && request.path === '/v1/models') {
      const body = `{"object":"list","data":[{"id":${JSON.stringify(this.modelId)},` +
        '"object":"model","owned_by":"local"}]}';
      await this.sendAndClose(client, this.httpResponse(200, 'OK', body, client.origin));
      return;
    }
    if (request.method === 'GET' && request.path === '/v1/queue') {
      const body = `{"queue_size":${GenerationCoordinator.getInstance().queueSize()}}`;
      await this.sendAndClose(client, this.httpResponse(200, 'OK', body, client.origin));
      return;
    }
    if (request.method === 'GET' && request.path === '/v1/queue/status') {
      // OpenAI-compatible queue status, matching the Android client's response shape.
      const stats = GenerationCoordinator.getInstance().getQueueStats();
      const message = stats.isProcessing ? '正在处理请求中' :
        (stats.pendingRequests > 0 ? `队列中有 ${stats.pendingRequests} 个待处理请求` : '队列空闲');
      const body = `{"status":"ok","totalRequests":${stats.totalRequests},` +
        `"completedRequests":${stats.completedRequests},"failedRequests":${stats.failedRequests},` +
        `"pendingRequests":${stats.pendingRequests},"isProcessing":${stats.isProcessing},` +
        `"successRate":${stats.successRate.toFixed(2)},"message":${JSON.stringify(message)}}`;
      await this.sendAndClose(client, this.httpResponse(200, 'OK', body, client.origin));
      return;
    }
    if (request.method === 'DELETE' && request.path === '/v1/queue/clear') {
      const cleared = GenerationCoordinator.getInstance().clearPendingQueue();
      await this.sendAndClose(client, this.httpResponse(200, 'OK',
        `{"status":"ok","message":"队列已清空","cleared":${cleared}}`, client.origin));
      return;
    }
    if (request.method === 'POST' && request.path === '/v1/cancel') {
      const cancel = JSON.parse(request.body) as CancelRequestDocument;
      if (!cancel.request_id) {
        await this.sendAndClose(client, this.errorResponse(400, 'request_id is required', client.origin));
        return;
      }
      GenerationCoordinator.getInstance().cancel(cancel.request_id);
      await this.sendAndClose(client, this.httpResponse(200, 'OK', '{"canceled":true}', client.origin));
      return;
    }
    if (request.method === 'POST' &&
      (request.path === '/v1/chat/completions' || request.path === '/v1/messages')) {
      if (!(request.headers.get('content-type') || '').toLowerCase().startsWith('application/json')) {
        await this.sendAndClose(client, this.errorResponse(415, 'content-type must be application/json', client.origin));
        return;
      }
      await this.generate(client, request, request.path === '/v1/messages');
      return;
    }
    await this.sendAndClose(client, this.errorResponse(404, 'not found', client.origin));
  }

  private async generate(client: ClientState, request: ParsedRequest, anthropic: boolean): Promise<void> {
    if (!this.target) {
      await this.sendAndClose(client, this.errorResponse(503, 'model is not ready', client.origin));
      return;
    }
    if (GenerationCoordinator.getInstance().queueSize() >= 8) {
      await this.sendAndClose(client, this.httpResponse(429, 'Too Many Requests',
        '{"error":{"message":"generation queue is full","type":"rate_limit_error"}}',
        client.origin, 'Retry-After: 1\r\n'));
      return;
    }
    const document = JSON.parse(request.body) as ChatRequestDocument;
    if (!document.messages || document.messages.length === 0) {
      await this.sendAndClose(client, this.errorResponse(400, 'messages is required', client.origin));
      return;
    }
    const messages: NativeChatMessage[] = [];
    if (document.system) {
      messages.push({ role: 'system', content: document.system });
    } else {
      messages.push({ role: 'system', content: this.systemPrompt });
    }
    document.messages.forEach((message: ApiMessageDocument) => {
      if ((message.role === 'user' || message.role === 'assistant' || message.role === 'system') &&
        typeof message.content === 'string') {
        messages.push({ role: message.role, content: message.content });
      }
    });
    if (messages.length < 2) {
      await this.sendAndClose(client, this.errorResponse(400, 'no valid messages', client.origin));
      return;
    }
    const requestId = GenerationCoordinator.getInstance().createRequestId('api');
    client.requestId = requestId;
    const generationRequest: NativeGenerateRequest = {
      requestId: requestId,
      messages: messages,
      maxTokens: Math.min(4096, Math.max(1, document.max_tokens || this.maxTokens))
    };
    let fullText = '';
    if (document.stream) {
      await this.queueSend(client, 'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n' +
        `${this.corsHeaders(client.origin)}` +
        'Cache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n');
      if (anthropic) {
        await this.queueSse(client,
          `event: message_start\ndata: {"type":"message_start","message":{"id":${JSON.stringify(requestId)},` +
            '"type":"message","role":"assistant","content":[],"stop_reason":null}}\n\n');
        await this.queueSse(client,
          'event: content_block_start\ndata: {"type":"content_block_start","index":0,' +
            '"content_block":{"type":"text","text":""}}\n\n');
      }
    }
    let result: NativeGenerateResult;
    try {
      result = await GenerationCoordinator.getInstance().generate(this.target, generationRequest, (chunk: string) => {
        fullText += chunk;
        if (document.stream) {
          const escapedChunk = JSON.stringify(chunk);
          const payload = anthropic ?
            `event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta",` +
              `"text":${escapedChunk}}}\n\n` :
            `data: {"id":${JSON.stringify(requestId)},"object":"chat.completion.chunk","choices":[{"index":0,` +
              `"delta":{"content":${escapedChunk}},"finish_reason":null}]}\n\n`;
          this.queueSse(client, payload);
        }
      });
    } catch (error) {
      if (!document.stream) {
        throw error as Error;
      }
      const message = JSON.stringify((error as Error).message);
      const payload = anthropic ?
        `event: error\ndata: {"type":"error","error":{"type":"api_error","message":${message}}}\n\n` :
        `data: {"error":{"message":${message},"type":"api_error"}}\n\n`;
      await this.queueSse(client, payload);
      if (!anthropic) {
        await this.queueSse(client, 'data: [DONE]\n\n');
      }
      await this.queueSend(client, '0\r\n\r\n');
      await client.writeTail;
      await client.connection.close();
      client.requestId = '';
      return;
    }
    if (document.stream) {
      if (anthropic) {
        await this.queueSse(client,
          'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n');
        await this.queueSse(client,
          `event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":` +
            `"${result.canceled ? 'stop_sequence' : 'end_turn'}"},"usage":{"output_tokens":` +
            `${result.generatedTokens}}}\n\n`);
      }
      const finalEvent = anthropic ? 'event: message_stop\ndata: {"type":"message_stop"}\n\n' :
        'data: [DONE]\n\n';
      await this.queueSse(client, finalEvent);
      await this.queueSend(client, '0\r\n\r\n');
      await client.writeTail;
      await client.connection.close();
    } else {
      const text = JSON.stringify(result.text || fullText);
      const response = anthropic ?
        `{"id":${JSON.stringify(requestId)},"type":"message","role":"assistant",` +
          `"content":[{"type":"text","text":${text}}],` +
          `"stop_reason":"${result.canceled ? 'stop_sequence' : 'end_turn'}",` +
          `"usage":{"input_tokens":${result.promptTokens},"output_tokens":${result.generatedTokens}}}` :
        `{"id":${JSON.stringify(requestId)},"object":"chat.completion","model":${JSON.stringify(this.modelId)},` +
          `"choices":[{"index":0,"message":{"role":"assistant","content":${text}},` +
          `"finish_reason":"stop"}],"usage":{"prompt_tokens":${result.promptTokens},` +
          `"completion_tokens":${result.generatedTokens},` +
          `"total_tokens":${result.promptTokens + result.generatedTokens}}}`;
      await this.sendAndClose(client, this.httpResponse(200, 'OK', response, client.origin));
    }
    client.requestId = '';
  }

  private authorized(request: ParsedRequest): boolean {
    if (!this.authEnabled) {
      return true;
    }
    const authorization = request.headers.get('authorization') || '';
    const bearer = authorization.startsWith('Bearer ') ? authorization.substring(7) : '';
    const apiKey = request.headers.get('x-api-key') || '';
    return this.constantTimeEqual(bearer || apiKey, this.token);
  }

  private constantTimeEqual(left: string, right: string): boolean {
    let difference = left.length ^ right.length;
    const length = Math.max(left.length, right.length);
    for (let index = 0; index < length; index++) {
      difference |= (index < left.length ? left.charCodeAt(index) : 0) ^
        (index < right.length ? right.charCodeAt(index) : 0);
    }
    return difference === 0;
  }

  private async ensureToken(): Promise<string> {
    let token = await SettingsStore.getInstance().getString(API_TOKEN_KEY);
    if (!token) {
      token = `${util.generateRandomUUID(false)}${util.generateRandomUUID(false)}`.replace(/-/g, '');
      await SettingsStore.getInstance().putString(API_TOKEN_KEY, token);
    }
    return token;
  }

  private append(first: Uint8Array, second: Uint8Array): Uint8Array {
    const result = new Uint8Array(first.byteLength + second.byteLength);
    result.set(first, 0);
    result.set(second, first.byteLength);
    return result;
  }

  private findHeaderEnd(buffer: Uint8Array): number {
    for (let index = 0; index <= buffer.byteLength - 4; index++) {
      if (buffer[index] === 13 && buffer[index + 1] === 10 && buffer[index + 2] === 13 &&
        buffer[index + 3] === 10) {
        return index;
      }
    }
    return -1;
  }

  private httpResponse(status: number, reason: string, body: string, origin: string = '',
    extraHeaders: string = ''): string {
    const length = util.TextEncoder.create('utf-8').encodeInto(body).byteLength;
    const cors = this.corsHeaders(origin);
    return `HTTP/1.1 ${status} ${reason}\r\nContent-Type: application/json; charset=utf-8\r\n` +
      `${cors}${extraHeaders}Content-Length: ${length}\r\nConnection: close\r\n\r\n${body}`;
  }

  /**
   * Builds CORS response headers. An empty allowlist permits every origin (mirroring the Android
   * client); otherwise only origins whose host matches the comma-separated allowlist are echoed.
   */
  private corsHeaders(origin: string): string {
    if (!this.corsEnabled) {
      return '';
    }
    const allowed = this.allowedOrigin(origin);
    if (!allowed) {
      return '';
    }
    return `Access-Control-Allow-Origin: ${allowed}\r\n` +
      'Access-Control-Allow-Headers: Authorization, Content-Type, x-api-key, anthropic-version\r\n' +
      'Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n';
  }

  private allowedOrigin(origin: string): string {
    const entries: string[] = [];
    this.corsOrigins.split(',').forEach((entry: string) => {
      const trimmed = entry.trim();
      if (trimmed) {
        entries.push(trimmed.replace(/^https?:\/\//, ''));
      }
    });
    if (entries.length === 0) {
      return '*';
    }
    if (!origin) {
      return '';
    }
    const host = origin.replace(/^https?:\/\//, '');
    for (let index = 0; index < entries.length; index++) {
      if (entries[index] === host) {
        return origin;
      }
    }
    return '';
  }

  private errorResponse(status: number, message: string, origin: string = ''): string {
    const body = `{"error":{"message":${JSON.stringify(message)},"type":"invalid_request_error"}}`;
    return this.httpResponse(status, status === 400 ? 'Bad Request' : 'Error', body, origin);
  }

  private queueSse(client: ClientState, payload: string): Promise<void> {
    const length = util.TextEncoder.create('utf-8').encodeInto(payload).byteLength;
    return this.queueSend(client, `${length.toString(16)}\r\n${payload}\r\n`);
  }

  private queueSend(client: ClientState, data: string): Promise<void> {
    client.writeTail = client.writeTail.then(() => client.connection.send({ data: data, encoding: 'UTF-8' }));
    return client.writeTail;
  }

  private sendRaw(connection: socket.TCPSocketConnection, data: string): Promise<void> {
    return connection.send({ data: data, encoding: 'UTF-8' });
  }

  private async sendAndClose(client: ClientState, data: string): Promise<void> {
    await this.queueSend(client, data);
    await client.connection.close();
    this.recordRequest(client, data);
  }

  private recordRequest(client: ClientState, data: string): void {
    if (!client.lastMethod) {
      return;
    }
    const match = data.match(/^HTTP\/1\.1 (\d{3})/);
    const log = new ApiRequestLog();
    log.time = new Date().toTimeString().substring(0, 8);
    log.method = client.lastMethod;
    log.path = client.lastPath;
    log.status = match ? Number(match[1]) : 0;
    log.durationMs = client.lastStartMs > 0 ? Date.now() - client.lastStartMs : 0;
    this.requestLogs.push(log);
    if (this.requestLogs.length > 20) {
      this.requestLogs.shift();
    }
    client.lastMethod = '';
    client.lastPath = '';
    client.lastStartMs = 0;
  }

  private closeClient(client: ClientState): void {
    if (client.requestId) {
      GenerationCoordinator.getInstance().cancel(client.requestId);
    }
    this.clients.delete(client.connection.clientId);
  }
}
