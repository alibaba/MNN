import numpy
import json
def write_quant_header(file, ic, oc, quant_bit):
    dim_num = file.write(b'\x02')
    shape_dtype = numpy.int16
    if oc > 65535 or ic > 65535:
        shape_dtype = numpy.int32
    dim_length = file.write(numpy.array([oc, ic]).astype(shape_dtype))
    offset = 1 << (quant_bit - 1)
    weight_map = [i for i in range(-offset, offset)]
    if len(weight_map) == 256:
        weight_map.insert(0, 0)
    else:
        weight_map.insert(0, len(weight_map))
    map_length = file.write(numpy.array(weight_map, dtype=numpy.int8))
    header_length = dim_num + dim_length + map_length
    return header_length, shape_dtype == numpy.int32


def repack_low_bits(x, iNeedBits, block_size):
    v = []
    block_number = x.shape[0]
    count = block_size * iNeedBits // 8
    for i in range(0, count):
        v.append(numpy.zeros([block_number, 1]).astype(numpy.uint8))
    iOffset = 0
    cMask = (1 << iNeedBits) - 1
    index = 0
    for i in range(0, block_size):
        p0 = x[:, i:i+1]
        uShift = 8 - iNeedBits - (iOffset % 8)
        if uShift < 0:
            v[index+iOffset // 8] |= ((p0 & cMask) >> (0 - uShift))
            v[index+(iOffset // 8) + 1] |= ((p0 & cMask) << (8 + uShift))
        else:
            v[index+iOffset // 8] |= ((p0 & cMask) << uShift)
        iOffset += iNeedBits
        if iOffset % 8 == 0:
            index += iOffset // 8
            iOffset = 0
    return numpy.concatenate(v, axis=1) 

class Block:
    def __init__(self):
        self.conv = []
        self.attn_norm = None
        self.ffn_norm = None

def load_mnn(filename):
    mnn = {}
    with open(filename) as f:
        mnn = json.load(f)
    block = None
    blockes = []
    conv_index = 0
    conv_order = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
    for op in mnn["oplists"]:
        if op['type'] == 'LayerNorm':
            if 'external' in op['main']:
                del op['main']['external']
            if 'gamma' in op['main']:
                del op['main']['gamma']
            if 'beta' in op['main']:
                del op['main']['beta']
            if None != block:
                # post layernorm
                block.ffn_norm = op
            else:
                # start layernorm
                block = Block()
                block.attn_norm = op
                conv_index = 0
        if op['type'] == 'Convolution':
            block.conv.append(op)
            conv_index += 1
            if conv_index == 7:
                # end block
                blockes.append(block)
                block = None

    # Last layernorm and lm
    output_norm = block.attn_norm
    lm = block.conv[0]
    lm['name'] = 'output'
    opmap = {}
    opmap['output_norm'] = output_norm
    convs = []
    for i in range(0, len(blockes)):
        _block = blockes[i]
        opmap['blk.%d' %i + '.attn_norm'] = _block.attn_norm
        opmap['blk.%d' %i + '.ffn_norm'] = _block.ffn_norm
        for j in range(0, 7):
            newname = 'blk.%d' %i + '.' + conv_order[j]
            _block.conv[j]['name'] = newname
            convs.append(_block.conv[j])
    convs.append(lm)

    return mnn, opmap, convs, blockes, block


def write_quant_parameters(quant_bit, asymc, mnn_weight_file, ic, oc, weight_main, scalebias, mnn_weight_offset, need_scale_treat = True):
    conv = {}
    aMin = 0
    readType = 0
    if asymc:
        # Avoid aMin post treat for bias
        offset = -(1 << (quant_bit - 1))
        aMin = 1
        if need_scale_treat:
            scalebias = scalebias.reshape([-1, 2])
            bias = scalebias[:, 0:1]
            scale = scalebias[:, 1:2]
            bias = bias - offset * scale
            scalebias = numpy.concatenate([bias, scale], axis=1).astype(numpy.float32)
        readType = 1
    header_len, shape_int32 = write_quant_header(mnn_weight_file, ic, oc, quant_bit)
    weight_len =  mnn_weight_file.write(weight_main.tobytes()) + header_len
    alpha_len = mnn_weight_file.write(scalebias.tobytes())
    conv['quanParameter'] = {
        "quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0,
        "useInt32": False, "has_scaleInt": False, "shapeInt32": shape_int32,
        "type": 1, "aMaxOrBits": quant_bit, "aMin": aMin, "readType": readType, "weightSize": 0
    }
    conv['external'] = [mnn_weight_offset, weight_len, alpha_len, oc * 4, 0]
    mnn_weight_offset += (weight_len + alpha_len)   
    return conv, header_len, mnn_weight_offset