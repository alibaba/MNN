#ifndef MNN_ATTENTION_EXECUTION_HPP
#define MNN_ATTENTION_EXECUTION_HPP

#include "../../../core/OpCommonUtils.hpp"
#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
// CUDA Kernel 参数, 类似于 Metal 的 Param 结构体
// 这个结构体将被复制到 GPU 供 Kernel 使用
struct AttentionKernelParam {
    int query_seq_len;    // L_q (完整查询序列长度)
    int q_seq_piece_len;  // L_q_piece (当前正在处理的查询序列片段)
    int key_seq_len;      // L_k_total (KV Cache中或当前步骤的Key/Value的总当前长度)
    int head_num;         // H_q (查询头数量)
    int kv_head_num;      // H_kv (Key/Value头数量)
    int group;            // H_q / H_kv
    int head_dim;         // D (每个头的维度)
    float scale;          // 缩放因子
    int max_kv_len;       // L_kv_max (为KV Cache分配的长度, 或为当前步骤的临时V分配的长度)
    int batch;            // B (批次大小)
    int current_kv_seq_len_new; // 本步骤中新增的K/V的长度 (来自 AttentionExecution 类中的 mNewKvSeqLen)
    int past_kv_len;      // mCache->mPastLength (在此步骤之前KV Cache中已有的K/V长度)
};


class AttentionExecution : public Execution {
public:
    struct SharedCache {
        std::shared_ptr<Tensor> mPastKey;   // 存储 K Cache, GPU Tensor. 布局: [L_kv_max, B, H_kv, D]
        std::shared_ptr<Tensor> mPastValue; // 存储 V Cache, GPU Tensor. 布局: [B, H_kv, D, L_kv_max]
        int mPastLength = 0;                // Cache中当前实际的token数量 (L_k_past)
        int mMaxLength = 0;                 // Cache的已分配容量 (L_kv_max)
    };

    AttentionExecution(Backend *backend, bool kv_cache_op_param); // kv_cache_op_param 来自 Op 定义
    virtual ~AttentionExecution();

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode init_cache_tensors(); // 如果需要，初始化 mPastKey/mPastValue为空Tensor
    ErrorCode reallocKVCache_gpu(int required_total_kv_len, int batch_size, int kv_num_head, int head_dim, cudaStream_t stream);
    ErrorCode reallocKVCache_gpu(int required_total_kv_len, const KVMeta* meta, cudaStream_t stream);
    ErrorCode ensureTempBuffers_gpu(int batch, int num_head, int q_seq_piece_len_max, int current_max_total_kv_len, int head_dim);

    CUDABackend* mCudaBackend;
    bool mIsKVCacheEnabled; // 基于 Op 参数
    std::shared_ptr<SharedCache> mCache;
    float mScale;

    // onResize/onExecute 中获取的参数
    int mBatch;
    int mQuerySeqLen;       // 当前输入的完整查询序列长度 (L_q)
    int mNumHead;           // 查询头数量 (H_q)
    int mHeadDim;           // 每个头的维度 (D)
    int mKvNumHead;         // KV头数量 (H_kv)
    int mNewKvSeqLen;       // 当前输入的K/V Tensor的序列长度 (L_k_new, 即将追加的长度)

    int mQseqSplitNum;      // 查询序列分割数量

    // GPU上的临时Tensor
    std::shared_ptr<Tensor> mTempQK;        // 存储 (Q*K^T)/sqrt(d)。形状: [B, H_q, Max_L_q_piece, Max_L_k_total]
    std::shared_ptr<Tensor> mTempSoftmax;   // 存储 Softmax 输出。形状: [B, H_q, Max_L_q_piece, Max_L_k_total]
    
    // 如果 !mIsKVCacheEnabled, 用于存储当前步骤的K/V (类Cache格式)
    std::shared_ptr<Tensor> mTempK_current_step; // 布局: [Max_L_k_new_alloc, B, H_kv, D]
    std::shared_ptr<Tensor> mTempV_current_step; // 布局: [B, H_kv, D, Max_L_k_new_alloc]

    // Mask 相关
    bool mHasMask;
    bool mIsAddMask; // 如果 mask 是 float 类型并相加则为 true, 如果是 int 类型并设为 -FLT_MAX 则为 false

    AttentionKernelParam* mParam_gpu = nullptr; // Kernel参数的设备指针
    KVMeta* mMeta = nullptr;

    const int mExpandChunk = 64; // KV Cache重分配时的扩展块大小
    halide_type_t mPrecision;    // 精度 (float或half)
};
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN

#endif // MNN_ATTENTION_EXECUTION_HPP