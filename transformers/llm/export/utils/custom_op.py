import torch

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
    def symbolic(g, query, key, value, attention_mask, hidden_size, name):
        # These become the operator attributes.
        kwargs = {
            "hidden_size_i": hidden_size,
            "name_s": name
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(query)
        output_type = query.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FusedAttention", query, key, value, attention_mask, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, hidden_size, name):
        out_shape = list(query.shape)[:2] + [hidden_size]
        return query.new_zeros(out_shape)

class FusedAttention(torch.nn.Module):
    def __init__(self, hidden_size, name):
        super(FusedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.name = name

    def forward(self, query, key, value, attention_mask):
        return FusedAttentionOp.apply(query, key, value, attention_mask, self.hidden_size, self.name)