import relationalStore from '@ohos.data.relationalStore';
import { AppDatabase } from './AppDatabase';
import { AttachmentRecord } from '../domain/AppTypes';

let attachmentSequence: number = 0;

export class AttachmentRepository {
  private static instance: AttachmentRepository = new AttachmentRepository();

  static getInstance(): AttachmentRepository {
    return AttachmentRepository.instance;
  }

  async save(messageId: string, attachments: AttachmentRecord[]): Promise<void> {
    const store = AppDatabase.getInstance().getStore();
    for (let index = 0; index < attachments.length; index++) {
      const attachment = attachments[index];
      if (!attachment.id) {
        attachment.id = `attachment-${Date.now()}-${++attachmentSequence}`;
      }
      attachment.messageId = messageId;
      if (!attachment.createdAt) {
        attachment.createdAt = Date.now();
      }
      const values: relationalStore.ValuesBucket = {
        id: attachment.id,
        message_id: attachment.messageId,
        media_type: attachment.mediaType,
        local_path: attachment.localPath,
        display_name: attachment.displayName,
        size_bytes: attachment.sizeBytes,
        created_at: attachment.createdAt
      };
      await store.insert('attachments', values);
    }
  }

  async getForMessage(messageId: string): Promise<AttachmentRecord[]> {
    const predicates = new relationalStore.RdbPredicates('attachments');
    predicates.equalTo('message_id', messageId).orderByAsc('created_at');
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    const attachments: AttachmentRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const attachment = new AttachmentRecord();
        attachment.id = result.getString(result.getColumnIndex('id'));
        attachment.messageId = result.getString(result.getColumnIndex('message_id'));
        attachment.mediaType = result.getString(result.getColumnIndex('media_type'));
        attachment.localPath = result.getString(result.getColumnIndex('local_path'));
        attachment.displayName = result.getString(result.getColumnIndex('display_name'));
        attachment.sizeBytes = result.getLong(result.getColumnIndex('size_bytes'));
        attachment.createdAt = result.getLong(result.getColumnIndex('created_at'));
        attachments.push(attachment);
      }
    } finally {
      result.close();
    }
    return attachments;
  }
}
