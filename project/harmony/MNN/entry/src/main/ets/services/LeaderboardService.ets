import { tr } from './I18n';
import { http } from '@kit.NetworkKit';
import { deviceInfo } from '@kit.BasicServicesKit';
import { hidebug } from '@kit.PerformanceAnalysisKit';
import { util } from '@kit.ArkTS';
import { SettingsStore } from './SettingsStore';

const LEADERBOARD_BASE_URL: string = 'https://mnn-mnnchatleaderboard.ms.show/gradio_api/call';
const LEADERBOARD_USER_ID_KEY: string = 'leaderboard_user_id';

interface SubmitPayload {
  user_id: string;
  llm_model: string;
  device_model: string;
  device_chipset: string;
  device_memory: string;
  prefill_speed: string;
  decode_speed: string;
  memory_usage: string;
}

interface GradioDataBody {
  data: string[];
}

interface RankRequestBody {
  data: string[];
}

interface GradioEventDocument {
  event_id?: string;
}

interface GradioRankItem {
  rank?: number;
  total_entries_in_model?: number;
  total?: number;
  prefill_speed?: number;
  score?: number;
}

export class RankData {
  rank: number = 0;
  totalUsers: number = 0;
  score: number = 0;
}

export class LeaderboardOutcome {
  success: boolean = false;
  message: string = '';
  rank: RankData = new RankData();
}

/**
 * Submits benchmark results to the public MNN Chat leaderboard and queries the user's rank,
 * mirroring the Android client's LeaderboardService (Gradio call API).
 */
export class LeaderboardService {
  private static instance: LeaderboardService = new LeaderboardService();

  static getInstance(): LeaderboardService {
    return LeaderboardService.instance;
  }

  async userId(): Promise<string> {
    const store = SettingsStore.getInstance();
    let id = await store.getString(LEADERBOARD_USER_ID_KEY, '');
    if (!id) {
      id = util.generateRandomUUID(false).replace(/-/g, '').substring(0, 16);
      await store.putString(LEADERBOARD_USER_ID_KEY, id);
    }
    return id;
  }

  async setUserId(id: string): Promise<void> {
    await SettingsStore.getInstance().putString(LEADERBOARD_USER_ID_KEY, id.trim());
  }

  async submit(modelName: string, prefillSpeed: number, decodeSpeed: number,
    memoryMb: number): Promise<LeaderboardOutcome> {
    const outcome = new LeaderboardOutcome();
    const payload: SubmitPayload = {
      user_id: await this.userId(),
      llm_model: modelName,
      device_model: deviceInfo.marketName || deviceInfo.productModel || 'HarmonyOS device',
      device_chipset: deviceInfo.hardwareModel || deviceInfo.brand || 'Unknown',
      device_memory: `${this.limitMemoryMb()} MB`,
      prefill_speed: prefillSpeed.toFixed(2),
      decode_speed: decodeSpeed.toFixed(2),
      memory_usage: memoryMb.toFixed(1)
    };
    const body: GradioDataBody = { data: [JSON.stringify(payload)] };
    try {
      await this.request('POST', `${LEADERBOARD_BASE_URL}/submit_score`, JSON.stringify(body));
      outcome.success = true;
      outcome.message = tr('已提交到排行榜');
    } catch (error) {
      outcome.success = false;
      outcome.message = (error as Error).message;
    }
    return outcome;
  }

  async queryRank(modelName: string): Promise<LeaderboardOutcome> {
    const outcome = new LeaderboardOutcome();
    const request: RankRequestBody = { data: [await this.userId(), modelName] };
    try {
      const text = await this.request('POST', `${LEADERBOARD_BASE_URL}/get_my_rank`, JSON.stringify(request));
      const eventId = this.extractEventId(text);
      if (!eventId) {
        outcome.message = tr('未获取到查询任务 ID');
        return outcome;
      }
      const result = await this.request('GET', `${LEADERBOARD_BASE_URL}/get_my_rank/${eventId}`, '');
      outcome.rank = this.parseRank(result);
      outcome.success = true;
      outcome.message = outcome.rank.rank > 0 ?
        `${tr('当前排名 ')}${outcome.rank.rank} / ${outcome.rank.totalUsers}` : tr('暂未进入排行榜');
    } catch (error) {
      outcome.success = false;
      outcome.message = (error as Error).message;
    }
    return outcome;
  }

  private async request(method: string, url: string, body: string): Promise<string> {
    const client = http.createHttp();
    try {
      const response = await client.request(url, {
        method: method === 'POST' ? http.RequestMethod.POST : http.RequestMethod.GET,
        header: {
          'Content-Type': 'application/json',
          'User-Agent': 'MNN-LLM-Chat HarmonyOS App'
        },
        extraData: body,
        connectTimeout: 30000,
        readTimeout: 30000,
        expectDataType: http.HttpDataType.STRING
      });
      const text = typeof response.result === 'string' ? response.result : '';
      if (response.responseCode < 200 || response.responseCode >= 300) {
        throw new Error(`HTTP ${response.responseCode}: ${text.substring(0, 120)}`);
      }
      return text;
    } finally {
      client.destroy();
    }
  }

  private extractEventId(text: string): string {
    try {
      const document = JSON.parse(text) as GradioEventDocument;
      return document.event_id || '';
    } catch (error) {
      return '';
    }
  }

  private parseRank(text: string): RankData {
    const rank = new RankData();
    const lines = text.split('\n');
    for (let index = 0; index < lines.length; index++) {
      const line = lines[index].trim();
      if (!line.startsWith('data:')) {
        continue;
      }
      const payload = line.substring(5).trim();
      if (!payload || payload === '[null]') {
        continue;
      }
      try {
        const array = JSON.parse(payload) as GradioRankItem[];
        if (array.length > 0 && array[0] !== null && array[0] !== undefined) {
          const item = array[0];
          rank.rank = item.rank ?? 0;
          rank.totalUsers = item.total_entries_in_model ?? item.total ?? 0;
          rank.score = item.prefill_speed ?? item.score ?? 0;
          return rank;
        }
      } catch (error) {
      }
    }
    return rank;
  }

  private limitMemoryMb(): number {
    try {
      const limit = hidebug.getAppMemoryLimit();
      if (limit.rssLimit > 0n) {
        return Number(limit.rssLimit / 1024n);
      }
    } catch (error) {
    }
    return 0;
  }
}