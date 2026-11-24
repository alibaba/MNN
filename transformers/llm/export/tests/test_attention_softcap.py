"""Offline CPU regressions for attention-score softcapping and export metadata."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EXPORT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPORT_DIR))
from utils.custom_op import FusedAttention
from utils.mnn_converter import MNNConverter
from utils.torch_utils import onnx_export
from utils.transformers import Attention


MISSING = object()
CAP = 'attn_logit_softcapping'
MAPPER = {'attention': {name: name for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')}}


def make_attention(attn_attrs=None, export_attrs=None):
    source = torch.nn.Module()
    source.q_proj = torch.nn.Linear(8, 4, bias=False)
    source.k_proj = torch.nn.Linear(8, 2, bias=False)
    source.v_proj = torch.nn.Linear(8, 2, bias=False)
    source.o_proj = torch.nn.Linear(4, 8, bias=False)
    with torch.no_grad():
        source.q_proj.weight.copy_(torch.eye(8)[:4])
        source.k_proj.weight.copy_(torch.eye(8)[4:6])
        source.v_proj.weight.copy_(torch.eye(8)[6:8])
        source.o_proj.weight.copy_(torch.cat((torch.eye(4), -0.5 * torch.eye(4))))
    source.scaling = 0.23
    for name, value in (attn_attrs or {}).items():
        setattr(source, name, value)
    config = SimpleNamespace(hidden_size=8, head_dim=2, num_attention_heads=2,
                             num_key_value_heads=1, model_type='gemma2')
    for name, value in (export_attrs or {}).items():
        setattr(config, name, value)
    return Attention(source, 0, config, None, MAPPER), source


def inputs_and_mask():
    hidden = torch.tensor([[[9., 3., -5., 2., 8., -2., 3., -1.],
                            [-7., 4., 6., -3., -9., 5., -2., 5.],
                            [3., -8., 2., 7., 4., 9., 9., 2.],
                            [11., -2., -8., -6., -6., -7., -7., -4.]]])
    mask = torch.zeros(4, 4).masked_fill(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1), -torch.inf)
    mask[1, 0], mask[3, 2] = -1.3, -2.0
    return hidden, mask[None, None]


def numeric_reference(source, hidden, mask, cap, order='correct'):
    # NumPy FP64 uses the original projections, independently of rebuilt Attention.
    x = hidden.detach().numpy()[0].astype(np.float64)
    projections = [x @ getattr(source, name).weight.detach().numpy().astype(np.float64).T
                   for name in ('q_proj', 'k_proj', 'v_proj')]
    query, key, value = projections
    query = query.reshape(x.shape[0], 2, 2)
    additive_mask = mask.numpy()[0, 0]
    outputs = []
    for head in range(2):
        scores = query[:, head] @ key.T
        if order == 'scale_after_cap':
            scores = cap * np.tanh(scores / cap) * source.scaling
        else:
            scores = scores * source.scaling
            if order == 'mask_before_cap':
                scores = scores + additive_mask
            if cap:
                scores = cap * np.tanh(scores / cap)
        if order != 'mask_before_cap':
            scores = scores + additive_mask
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        outputs.append(probabilities @ value)
    output = np.stack(outputs, axis=1).reshape(x.shape[0], 4)
    return (output @ source.o_proj.weight.detach().numpy().T)[None]


def test_eager_caps_scaled_scores_before_additive_mask():
    model, source = make_attention({'config': SimpleNamespace(**{CAP: 3.5})})
    hidden, mask = inputs_and_mask()
    actual = model(hidden, mask).detach().numpy()
    expected = numeric_reference(source, hidden, mask, 3.5)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    for cap, order in ((0., 'correct'), (3.5, 'mask_before_cap'), (3.5, 'scale_after_cap')):
        wrong = numeric_reference(source, hidden, mask, cap, order)
        assert np.max(np.abs(expected - wrong)) > 0.5
    assert source.scaling != pytest.approx(2 ** -0.5)


def test_eager_decode_preserves_kv_cache():
    model, source = make_attention({CAP: 3.5})
    hidden, mask = inputs_and_mask()
    model(hidden[:, :3], mask[:, :, :3, :3])
    actual = model(hidden[:, 3:], mask[:, :, 3:]).detach().numpy()
    expected = numeric_reference(source, hidden, mask, 3.5)[:, 3:]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert model.past_key_value.shape == (2, 1, 4, 1, 2)
    assert model.fused_attn.kv_cache == 1


@pytest.mark.parametrize('cap', [MISSING, None, 0.0], ids=['missing', 'none', 'zero'])
def test_disabled_cap_keeps_eager_attention_unchanged(cap):
    attrs = {} if cap is MISSING else {'config': SimpleNamespace(**{CAP: cap})}
    model, source = make_attention(attrs)
    hidden, mask = inputs_and_mask()
    actual = model(hidden, mask).detach().numpy()
    np.testing.assert_allclose(actual, numeric_reference(source, hidden, mask, 0.), rtol=2e-6, atol=2e-6)
    assert model.attn_logit_softcapping == 0.0
    assert model.fused_attn.softcap == 0.0


@pytest.mark.parametrize('attn_attrs,export_attrs,expected', [
    ({CAP: 3.5, 'config': SimpleNamespace(**{CAP: 5.})},
     {CAP: 99., 'origin_config': SimpleNamespace(**{CAP: 6.})}, 3.5),
    ({'config': SimpleNamespace(**{CAP: 5.})}, {CAP: 99.}, 5.),
    ({'config': SimpleNamespace(text_config=SimpleNamespace(**{CAP: 4.}), **{CAP: 5.})},
     {CAP: 99.}, 4.),
    ({}, {'origin_config': SimpleNamespace(**{CAP: 6.}), CAP: 99.}, 6.),
    ({}, {'origin_config': SimpleNamespace(text_config=SimpleNamespace(**{CAP: 7.}), **{CAP: 6.}),
          CAP: 99.}, 7.),
    ({}, {'origin_config': SimpleNamespace(text_config=SimpleNamespace(**{CAP: 7.}))}, 7.),
    ({}, {CAP: 8.}, 8.),
    ({}, {'text_config': SimpleNamespace(**{CAP: 9.})}, 9.),
    ({CAP: None}, {'origin_config': SimpleNamespace(**{CAP: 7.})}, 0.),
    ({'config': SimpleNamespace(**{CAP: None})}, {CAP: 99.}, 0.),
    ({CAP: 0.}, {'origin_config': SimpleNamespace(**{CAP: 7.})}, 0.),
    ({}, {'origin_config': SimpleNamespace(text_config=SimpleNamespace(**{CAP: None}), **{CAP: 7.})}, 0.),
])
def test_original_source_config_precedence(attn_attrs, export_attrs, expected):
    model, _ = make_attention(attn_attrs, export_attrs)
    assert model.attn_logit_softcapping == expected
    assert model.fused_attn.softcap == expected
    assert isinstance(model.fused_attn.softcap, float)


@pytest.mark.parametrize('cap', [-1., float('nan'), float('inf'), -float('inf'), 'invalid'])
@pytest.mark.parametrize('location', ['module', 'source_config', 'origin_text_config', 'mapped_config'])
def test_invalid_external_cap_is_rejected(cap, location):
    attn_attrs, export_attrs = {}, {}
    if location == 'module':
        attn_attrs[CAP] = cap
    elif location == 'source_config':
        attn_attrs['config'] = SimpleNamespace(**{CAP: cap})
    elif location == 'origin_text_config':
        export_attrs['origin_config'] = SimpleNamespace(text_config=SimpleNamespace(**{CAP: cap}))
    else:
        export_attrs[CAP] = cap
    with pytest.raises(ValueError, match='attn_logit_softcapping.*finite.*non-negative'):
        make_attention(attn_attrs, export_attrs)


def test_installed_hf_gemma2_attention_propagates_original_config():
    gemma2 = pytest.importorskip('transformers.models.gemma2.modeling_gemma2')
    config = gemma2.Gemma2Config(hidden_size=8, head_dim=2, num_attention_heads=2,
                               num_key_value_heads=1, num_hidden_layers=1, intermediate_size=16,
                               query_pre_attn_scalar=19, attn_logit_softcapping=3.5)
    source = gemma2.Gemma2Attention(config, layer_idx=0)
    mapped = SimpleNamespace(hidden_size=8, head_dim=2, num_attention_heads=2,
                             num_key_value_heads=1, model_type='gemma2', origin_config=config)
    model = Attention(source, 0, mapped, None, MAPPER)
    assert model.attn_logit_softcapping == model.fused_attn.softcap == 3.5
    assert model.attn_scaling == pytest.approx(19 ** -0.5)


@pytest.mark.parametrize('cap', [MISSING, 0., 3.5], ids=['legacy-default', 'zero', 'positive'])
@pytest.mark.parametrize('with_sink', [False, True])
def test_actual_fused_onnx_symbolic_float_attr_preserves_tensor_inputs(cap, with_sink):
    onnx = pytest.importorskip('onnx')
    args = (4, True, '/attention', 7, 3, 2)
    module = FusedAttention(*args) if cap is MISSING else FusedAttention(*args, cap)
    query = torch.zeros(1, 3, 2, 2)
    key = torch.zeros(1, 3, 1, 2)
    inputs = (query, key, key.clone(), torch.zeros(1, 1, 3, 3))
    names = ['query', 'key', 'value', 'attention_mask']
    if with_sink:
        inputs += (torch.tensor([0.1, -0.3]),)
        names += ['attention_sink']
    buffer = io.BytesIO()
    onnx_export(module.eval(), inputs, buffer, names, ['output'])
    graph = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    node, = graph.graph.node
    assert node.op_type == 'FusedAttention' and node.domain == 'LlmExporter'
    assert list(node.input) == names
    assert len(graph.graph.input) == (5 if with_sink else 4)
    attrs = {attr.name: attr for attr in node.attribute}
    assert attrs['softcap'].type == onnx.AttributeProto.FLOAT
    assert attrs['softcap'].f == pytest.approx(0. if cap is MISSING else cap)
    assert {name: attrs[name].i for name in ('kv_cache', 'layer_index', 'kv_shared_layer_index', 'head_dim')} == {
        'kv_cache': 1, 'layer_index': 7, 'kv_shared_layer_index': 3, 'head_dim': 2}
    assert graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 4


def test_attention_onnx_retains_fused_op_and_cache():
    onnx = pytest.importorskip('onnx')
    module, _ = make_attention(export_attrs={
        'origin_config': SimpleNamespace(text_config=SimpleNamespace(**{CAP: 3.5}))})
    module.export_fused_attn = True
    hidden, mask = inputs_and_mask()
    buffer = io.BytesIO()
    onnx_export(module.eval(), (hidden, mask), buffer, ['hidden', 'mask'], ['output'])
    graph = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    node, = [node for node in graph.graph.node if node.op_type == 'FusedAttention']
    attrs = {attr.name: attr for attr in node.attribute}
    assert attrs['softcap'].type == onnx.AttributeProto.FLOAT
    assert attrs['softcap'].f == 3.5
    assert attrs['kv_cache'].i == 1
    assert len(node.input) == 4
    assert not any(node.op_type == 'Tanh' for node in graph.graph.node)


@pytest.mark.parametrize('cap', [MISSING, 0., 3.5], ids=['missing', 'zero', 'positive'])
@pytest.mark.parametrize('input_count', [4, 5])
def test_converter_reconstruction_preserves_softcap_and_defaults(cap, input_count):
    converter = MNNConverter(SimpleNamespace(args=SimpleNamespace(mnnconvert='')))
    attrs = [{'key': 'name', 's': '/attention'}, {'key': 'kv_cache', 'i': 1},
             {'key': 'layer_index', 'i': 7}, {'key': 'kv_shared_layer_index', 'i': 3},
             {'key': 'head_dim', 'i': 2}, {'key': 'output_dim', 'i': 4}]
    if cap is not MISSING:
        attrs.append({'key': 'softcap', 'f': cap})
    op = {'main': {'attr': attrs}, 'inputIndexes': list(range(input_count)), 'outputIndexes': [9]}
    rebuilt = converter.rebuild_attnention(op, {})
    assert rebuilt == [{'inputIndexes': op['inputIndexes'], 'main_type': 'AttentionParam',
                        'main': {'kv_cache': True, 'layer_index': 7, 'kv_shared_layer_index': 3,
                                 'softcap': 0. if cap is MISSING else cap},
                        'name': '/attention', 'outputIndexes': [9], 'type': 'Attention',
                        'defaultDimentionFormat': 'NHWC'}]


def test_converter_legacy_layer_defaults_are_unchanged():
    converter = MNNConverter(SimpleNamespace(args=SimpleNamespace(mnnconvert='')))
    op = {'main': {'attr': [{'key': 'name', 's': '/legacy'}, {'key': 'kv_cache', 'i': 0}]},
          'inputIndexes': [0, 1, 2, 3], 'outputIndexes': [4]}
    node, = converter.rebuild_attnention(op, {})
    assert node['main'] == {'kv_cache': False, 'layer_index': -1, 'kv_shared_layer_index': -1, 'softcap': 0.}
