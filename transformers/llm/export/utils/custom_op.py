
import torch
import torch.nn.functional as F

class FakeLinearOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, has_bias, name):
        # These become the operator attributes.
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "has_bias_i": has_bias,
            "name_s": name
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FakeLinear", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, has_bias, name):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class FakeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, name):
        super(FakeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.name = name

    def forward(self, x):
        return FakeLinearOp.apply(x, self.in_features, self.out_features, self.has_bias, self.name)

class FusedAttentionOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index,
                 kv_shared_layer_index, head_dim):
        # These become the operator attributes.
        kwargs = {
            "output_dim_i": output_dim,
            "kv_cache_i": kv_cache,
            "name_s": name,
            "layer_index_i": layer_index,
            "kv_shared_layer_index_i": kv_shared_layer_index,
            "head_dim_i": head_dim,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(query)
        out_sizes[-1] = output_dim
        output_type = query.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FusedAttention", query, key, value, attention_mask, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index,
                kv_shared_layer_index, head_dim):
        out_shape = list(query.shape)[:2] + [output_dim]
        return query.new_zeros(out_shape)

class FusedAttention(torch.nn.Module):
    def __init__(self, hidden_size, kv_cache, name, layer_index=-1, kv_shared_layer_index=-1,
                 head_dim=0):
        super(FusedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.kv_cache = int(kv_cache)
        self.name = name
        self.layer_index = layer_index
        self.kv_shared_layer_index = kv_shared_layer_index
        self.head_dim = int(head_dim)

    def forward(self, query, key, value, attention_mask):
        return FusedAttentionOp.apply(
            query, key, value, attention_mask, self.hidden_size, self.kv_cache, self.name,
            self.layer_index, self.kv_shared_layer_index, self.head_dim)

class FusedRoPEOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, cos, sin, q_norm_weight, k_norm_weight, rope_cut_head_dim, num_head,
                 kv_num_head, head_dim, q_norm_eps, k_norm_eps, q_norm, k_norm, name):
        kwargs = {
            "rope_cut_head_dim_i": rope_cut_head_dim,
            "num_head_i": num_head,
            "kv_num_head_i": kv_num_head,
            "head_dim_i": head_dim,
            "q_norm_eps_f": q_norm_eps,
            "k_norm_eps_f": k_norm_eps,
            "q_norm_i": q_norm,
            "k_norm_i": k_norm,
            "name_s": name,
        }
        query_output, key_output = g.op(
            "LlmExporter::FusedRoPE",
            query,
            key,
            cos,
            sin,
            q_norm_weight,
            k_norm_weight,
            **kwargs,
            outputs=2,
        )
        query_output.setType(query.type())
        key_output.setType(key.type())
        return query_output, key_output

    @staticmethod
    def forward(ctx, query, key, cos, sin, q_norm_weight, k_norm_weight, rope_cut_head_dim, num_head,
                kv_num_head, head_dim, q_norm_eps, k_norm_eps, q_norm, k_norm, name):
        return query, key

class FusedRoPE(torch.nn.Module):
    def __init__(self, rope_cut_head_dim, num_head, kv_num_head, head_dim, name):
        super(FusedRoPE, self).__init__()
        self.rope_cut_head_dim = int(rope_cut_head_dim)
        self.num_head = int(num_head)
        self.kv_num_head = int(kv_num_head)
        self.head_dim = int(head_dim)
        self.name = name

    @staticmethod
    def norm_eps(norm):
        if hasattr(norm, 'variance_epsilon'):
            return float(norm.variance_epsilon)
        return float(norm.eps)

    def forward(self, query, key, cos, sin, q_norm=None, k_norm=None):
        q_norm_weight = query.new_empty((0,)) if q_norm is None else q_norm.weight
        k_norm_weight = key.new_empty((0,)) if k_norm is None else k_norm.weight
        q_norm_eps = 0.0 if q_norm is None else self.norm_eps(q_norm)
        k_norm_eps = 0.0 if k_norm is None else self.norm_eps(k_norm)
        return FusedRoPEOp.apply(
            query,
            key,
            cos,
            sin,
            q_norm_weight,
            k_norm_weight,
            self.rope_cut_head_dim,
            self.num_head,
            self.kv_num_head,
            self.head_dim,
            q_norm_eps,
            k_norm_eps,
            int(q_norm is not None),
            int(k_norm is not None),
            self.name,
        )

class MoEOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, hidden_states, routing_weights, selected_experts, num_experts, top_k, layer_id):
        kwargs = {
            "num_experts_i": num_experts,
            "top_k_i": top_k,
            "layer_id_i": layer_id
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(hidden_states)
        output_type = hidden_states.type().with_sizes(out_sizes)
        return g.op("LlmExporter::MoE", hidden_states, routing_weights, selected_experts, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, hidden_states, routing_weights, selected_experts, num_experts, top_k, layer_id):
        return hidden_states

class MoE(torch.nn.Module):
    def __init__(self, num_experts, top_k, layer_id):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_id = layer_id

    def forward(self, hidden_states, routing_weights, selected_experts):
        return MoEOp.apply(hidden_states, routing_weights, selected_experts, self.num_experts, self.top_k, self.layer_id)

class FusedLinearAttentionOp(torch.autograd.Function):
    """
    Unified Custom Op for Linear Attention variants.

    Inputs (Tensors):
      0: qkv          [B, D, L]   - QKV projection output (before conv)
      1: gate         [B, L, H]   - Pre-computed decay factor
      2: beta         [B, L, H]   - Pre-computed learning rate (optional)
      3: conv_weight  [D, 1, K]   - Causal conv weight (optional)
      4: conv_bias    [D]         - Causal conv bias (optional)

    Attributes:
      type:             "gated_delta_rule" | "mamba" | "rwkv" | "gla" | "retnet"
      key_dim:          K total dim = num_k_heads * head_k_dim
      value_dim:        V total dim = num_v_heads * head_v_dim
      num_k_heads:      number of K/Q heads
      num_v_heads:      number of V heads (may differ from K for GQA)
      head_k_dim:       per-head K dimension
      head_v_dim:       per-head V dimension
      use_qk_l2norm:    whether to L2-normalize Q and K

    Output:
      0: attn_out  [B, L, num_v_heads, head_v_dim]

    Internal State (managed by MNN Execution, not in graph):
      conv_state:  [B, D, kernel_size - 1]
      rnn_state:   [B, num_v_heads, head_k_dim, head_v_dim]
    """
    @staticmethod
    def symbolic(g, qkv, gate, beta, conv_weight, name, attn_type,
                 num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        kwargs = {
            "name_s": name,
            "attn_type_s": attn_type,
            "num_k_heads_i": num_k_heads,
            "num_v_heads_i": num_v_heads,
            "head_k_dim_i": head_k_dim,
            "head_v_dim_i": head_v_dim,
            "use_qk_l2norm_i": int(use_qk_l2norm)
        }
        inputs = [qkv, gate, beta, conv_weight]
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        qkv_sizes = _get_tensor_sizes(qkv)
        # qkv shape is [Batch, Dim, SeqLen]
        batch_size = qkv_sizes[0]
        seq_len = qkv_sizes[2]

        out_sizes = [batch_size, seq_len, num_v_heads, head_v_dim]
        output_type = qkv.type().with_sizes(out_sizes)

        return g.op("LlmExporter::FusedLinearAttention", *inputs, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, qkv, gate, beta, conv_weight, name, attn_type,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        # Dummy forward: return correct output shape
        # qkv: [B, D, L] -> output: [B, L, num_v_heads, head_v_dim]
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        return qkv.new_zeros([batch_size, seq_len, num_v_heads, head_v_dim])

class FusedLinearAttention(torch.nn.Module):
    def __init__(self, name, attn_type, num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        super(FusedLinearAttention, self).__init__()
        self.name = name
        self.attn_type = attn_type
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, qkv, gate, beta, conv_weight):
        return FusedLinearAttentionOp.apply(qkv, gate, beta, conv_weight, self.name,
                                      self.attn_type, self.num_k_heads, self.num_v_heads,
                                      self.head_k_dim, self.head_v_dim, self.use_qk_l2norm)
