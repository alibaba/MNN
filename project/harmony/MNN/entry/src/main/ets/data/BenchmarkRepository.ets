import relationalStore from '@ohos.data.relationalStore';
import { AppDatabase } from './AppDatabase';
import { BenchmarkRecord } from '../domain/AppTypes';

let benchmarkSequence: number = 0;

export class BenchmarkRepository {
  private static instance: BenchmarkRepository = new BenchmarkRepository();

  static getInstance(): BenchmarkRepository {
    return BenchmarkRepository.instance;
  }

  async save(modelId: string, configSnapshot: string, resultJson: string): Promise<BenchmarkRecord> {
    const record = new BenchmarkRecord();
    record.id = `benchmark-${Date.now()}-${++benchmarkSequence}`;
    record.modelId = modelId;
    record.scenario = 'pp128_tg128';
    record.configSnapshot = configSnapshot;
    record.resultJson = resultJson;
    record.createdAt = Date.now();
    const values: relationalStore.ValuesBucket = {
      id: record.id,
      model_id: record.modelId,
      scenario: record.scenario,
      config_snapshot: record.configSnapshot,
      result_json: record.resultJson,
      created_at: record.createdAt
    };
    await AppDatabase.getInstance().getStore().insert('benchmark_results', values);
    return record;
  }

  async listLatest(modelId: string, limit: number = 10): Promise<BenchmarkRecord[]> {
    const predicates = new relationalStore.RdbPredicates('benchmark_results');
    predicates.equalTo('model_id', modelId).orderByDesc('created_at').limitAs(limit);
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    const records: BenchmarkRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const record = new BenchmarkRecord();
        record.id = result.getString(result.getColumnIndex('id'));
        record.modelId = result.getString(result.getColumnIndex('model_id'));
        record.scenario = result.getString(result.getColumnIndex('scenario'));
        record.configSnapshot = result.getString(result.getColumnIndex('config_snapshot'));
        record.resultJson = result.getString(result.getColumnIndex('result_json'));
        record.createdAt = result.getLong(result.getColumnIndex('created_at'));
        records.push(record);
      }
    } finally {
      result.close();
    }
    return records;
  }
}
