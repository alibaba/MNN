import os
from gguf import gguf_reader
from gguf import constants
import numpy
import json
import argparse

from utils.mnn_utils import *

class TokenContent:
    def __init__(self):
        self.token_type = -1
        self.spec_ids = []
        self.names = []
        self.stop_ids = []
        self.pre_ids = []
        self.token_num = 0

def load_token(reader):
    content = TokenContent()
    model = reader.fields['tokenizer.ggml.model'].parts[4].tobytes().decode('utf-8')
    field = reader.fields['tokenizer.ggml.token_type']
    valids = []
    for i in range(0, len(field.data)):
        p = field.data[i]
        if field.parts[p] == 1:
            #normal
            valids.append(i)
        elif field.parts[p] == 3 or field.parts[p] == 4:
            valids.append(i)
            content.spec_ids.append(i)
    tokens = reader.fields['tokenizer.ggml.tokens']
    stopes = ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>", "<EOT>"]

    for i in valids:
        p = tokens.data[i]
        tok = tokens.parts[p].tobytes().decode('utf-8')
        if tok in stopes:
            content.stop_ids.append(i)
        content.names.append(tok)
    content.token_num = len(content.names)
    if model == "gpt2":
        # bpe -> HUGGINGFACE
        content.token_type = 3
        # load merge
        merges = reader.fields['tokenizer.ggml.merges']
        for i in range(0, len(merges.data)):
            p = merges.data[i]
            tok = merges.parts[p].tobytes().decode('utf-8')
            content.names.append(tok)
    elif model == 'llama':
        content.token_type = 1
    else:
        print("[Error] Not support token type: , you can try download tokenizer.txt from old MNN LLM model", model)
    return content

def write_token_file(filename, token):
    with open(filename, 'w') as f:
        f.write("430 %d\n" %token.token_type)
        f.write("%d " %(len(token.spec_ids)) + '%d 0\n' %(len(token.stop_ids)))
        l = ""
        for i in token.spec_ids:
            l += "%d " %i
        for i in token.stop_ids:
            l += "%d " %i
        l+='\n'
        f.write(l)
        if token.token_type == 3:
            merge_num = len(token.names) - token.token_num
            f.write("%d " %token.token_num +  "%d\n" %merge_num)
        else:
            f.write("%d\n" %token.token_num)
        for name in token.names:
            f.write(name + '\n')
    return

def shuffle_weight_int4(weight_main):
    # shuffle weight
    block_number = weight_main.shape[0]
    half_block_size = weight_main.shape[1]
    weight_main_low = weight_main % 16
    weight_main_high = weight_main // 16
    weight_main =  numpy.concatenate([weight_main_low, weight_main_high], axis = 1).reshape([block_number, half_block_size, 2])
    weight_main_low = weight_main[:, :, 1]
    weight_main_high = weight_main[:, :, 0]
    weight_main = weight_main_low + weight_main_high * 16
    return weight_main

# const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
# const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

# const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
# const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

# y[i*qk + j + 0   ] = x0*d;
# y[i*qk + j + qk/2] = x1*d;

def shuffle_weight_int5(weight, repack = True):
    block_number = weight.shape[0]
    qh = weight[:, 0:4]
    qs = weight[:, 4:20]
    x0 = qs & 0x0F
    x1 = qs >> 4
    qh = numpy.frombuffer(qh.tobytes(), numpy.uint32).reshape([block_number, 1])
    mask_0 = []
    mask_1 = []
    for i in range(0, 16):
        mask_0.append(((qh >> i)<< 4) & 0x10)
        mask_1.append(((qh >> (i+12))) & 0x10)
    mask_0 = numpy.concatenate(mask_0, axis=1)
    mask_1 = numpy.concatenate(mask_1, axis=1)
    x0 = x0 + mask_0
    x1 = x1 + mask_1
    x = numpy.concatenate([x0, x1], axis=1)
    if repack:
        return repack_low_bits(x, 5, 32)
    return x

def extract_tensor_as_int8(weight):
    ic = int(weight.shape[0])
    oc = int(weight.shape[1])
    if weight.tensor_type == constants.GGMLQuantizationType.Q6_K:
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        scale_int8 = weight[:, 192:208]
        scale_half = weight[:, 208:210]
        scale_int8 = numpy.frombuffer(scale_int8.tobytes(), numpy.int8).astype(numpy.float32).reshape([block_number, 16, 1])
        scale_half = numpy.frombuffer(scale_half.tobytes(), numpy.float16).astype(numpy.float32).reshape([block_number, 1, 1])
        weight_scale = scale_half * scale_int8

        # Extract to int8
        ql = weight[:, 0:128]
        qh = weight[:, 128:192]

        qall = []
        for i in range(256):
            qall.append(None)
        for nnp in range(0, 2):
            for l in range(0, 32):
                q1 = ((ql[:, l +  0 + 64 * nnp] & 0xF) | (((qh[:, l + 32*nnp] >> 0) & 3) << 4))
                q2 = ((ql[:, l + 32 + 64 * nnp] & 0xF) | (((qh[:, l + 32*nnp] >> 2) & 3) << 4))
                q3 = ((ql[:, l +  0 + 64 * nnp]  >> 4) | (((qh[:, l + 32*nnp] >> 4) & 3) << 4))
                q4 = ((ql[:, l + 32 + 64 * nnp]  >> 4) | (((qh[:, l + 32*nnp] >> 6) & 3) << 4))
                qall[l + 0 + 128 * nnp] = q1.reshape([block_number, 1])
                qall[l + 32 + 128 * nnp] = q2.reshape([block_number, 1])
                qall[l + 64 + 128 * nnp] = q3.reshape([block_number, 1])
                qall[l + 96 + 128 * nnp] = q4.reshape([block_number, 1])
        q_raw = numpy.concatenate(qall, axis = 1)
        return q_raw, weight_scale, 16, 6
    elif weight.tensor_type == constants.GGMLQuantizationType.Q5_0:
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 2:type_size]
        weight_main = shuffle_weight_int5(weight_main, False)
        weight_scale = weight[:, 0:2]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).astype(numpy.float32)
        return weight_main, weight_scale, 32, 5
    return None

def write_external_weight(weight, mnn_weight_file, mnn_weight_offset):
    ic = int(weight.shape[0])
    oc = int(weight.shape[1])
    bias_length = oc * 4
    conv = {}
    block_size = 0
    block_number = 0
    quant_bit = 0
    tie_embedding = False
    header_len = 0
    if weight.tensor_type == constants.GGMLQuantizationType.F16:
        # FP16
        quan = {}
        quan['type'] = 3
        conv['quanParameter'] = quan
        rawbytes = weight.data.tobytes()
        weightlen = mnn_weight_file.write(rawbytes)
        external = [mnn_weight_offset, weightlen, 0, bias_length, 0]
        conv['external'] = external
        mnn_weight_offset += weightlen
        tie_embedding = True
        quant_bit = 16
    elif weight.tensor_type == constants.GGMLQuantizationType.F32:
        # FP16
        quan = {}
        quan['type'] = 3
        conv['quanParameter'] = quan
        rawbytes = weight.data.astype(numpy.float16).tobytes()
        weightlen = mnn_weight_file.write(rawbytes)
        external = [mnn_weight_offset, weightlen, 0, bias_length, 0]
        conv['external'] = external
        mnn_weight_offset += weightlen
    elif weight.tensor_type == constants.GGMLQuantizationType.Q4_0:
        tie_embedding = True
        quant_bit = 4
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        weight = weight.data.reshape([block_number, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 2:type_size]
        weight_scale = weight[:, 0:2]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).astype(numpy.float32)

        # shuffle weight
        weight_main = shuffle_weight_int4(weight_main)
        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, False, mnn_weight_file, ic, oc, weight_main, weight_scale, mnn_weight_offset)
    elif weight.tensor_type == constants.GGMLQuantizationType.Q4_1:
        quant_bit = 4
        tie_embedding = True
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 4:type_size]

        # shuffle weight
        weight_main = shuffle_weight_int4(weight_main);

        weight_scale = weight[:, 0:2]
        weight_bias = weight[:, 2:4]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).reshape((block_number, 1))
        weight_bias = numpy.frombuffer(weight_bias.tobytes(), numpy.float16).reshape((block_number, 1))
        scalebias = numpy.concatenate((weight_bias, weight_scale), axis=1).astype(numpy.float32)

        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, True, mnn_weight_file, ic, oc, weight_main, scalebias, mnn_weight_offset)
    elif weight.tensor_type == constants.GGMLQuantizationType.Q4_K:
        quant_bit = 4
        tie_embedding = True
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        d = weight[:, 0:2]
        dmin = weight[:, 2:4]
        scales = weight[:, 4:16]
        weight_main = weight[:, 16:type_size]

        # shuffle weight
        weight_main = weight_main.reshape((block_number * 4, 32))
        weight_main = shuffle_weight_int4(weight_main)

        # Compute Scale
        d = numpy.frombuffer(d.tobytes(), numpy.float16).reshape((block_number, 1)).astype(numpy.float32)
        dmin = numpy.frombuffer(dmin.tobytes(), numpy.float16).reshape((block_number, 1)).astype(numpy.float32)

        def get_scale_min_k4(j, q):
            if j < 4:
                d = q[:, j] & 63
                m = q[:, j + 4] & 63
            else:
                d = (q[:, j+4] & 0xF) | ((q[:, j-4] >> 6) << 4)
                m = (q[:, j+4] >>  4) | ((q[:, j-0] >> 6) << 4)
            return d, m
        dgroup=[]
        mgroup=[]
        for j in range(0, 8):
            dgroup.append(None)
            mgroup.append(None)
        for j in range(0, 8):
            vd, vm = get_scale_min_k4(j, scales)
            vd = vd.reshape((block_number, 1))
            vm = vm.reshape((block_number, 1))
            vd = vd.astype(numpy.float32) * d
            vm = vm.astype(numpy.float32) * dmin
            dgroup[j] = vd
            mgroup[j] = -vm
        weight_scale = numpy.concatenate(dgroup, -1).reshape((block_number, 8, 1))
        weight_bias = numpy.concatenate(mgroup, -1).reshape((block_number, 8, 1))
        scalebias = numpy.concatenate((weight_bias, weight_scale), axis=-1).astype(numpy.float32)


        block_size = 32

        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, True, mnn_weight_file, ic, oc, weight_main, scalebias, mnn_weight_offset)

    elif weight.tensor_type == constants.GGMLQuantizationType.Q8_0:
        quant_bit = 8
        tie_embedding = True
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 2:type_size]
        weight_scale = weight[:, 0:2]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).astype(numpy.float32)
        weight_main = numpy.frombuffer(weight_main.tobytes(), numpy.int8).astype(numpy.int16) + 128
        weight_main = weight_main.astype(numpy.uint8)
        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, False, mnn_weight_file, ic, oc, weight_main, weight_scale, mnn_weight_offset)

    elif weight.tensor_type == constants.GGMLQuantizationType.Q5_0:
        tie_embedding = False
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 2:type_size]
        weight_main = shuffle_weight_int5(weight_main)
        weight_scale = weight[:, 0:2]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).astype(numpy.float32)
        quant_bit = 5
        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, False, mnn_weight_file, ic, oc, weight_main, weight_scale, mnn_weight_offset)

    elif weight.tensor_type == constants.GGMLQuantizationType.Q5_1:
        tie_embedding = False
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        weight = weight.data.reshape([oc * ic // block_size, type_size])
        # Seperate Scale and Bias
        weight_main = weight[:, 4:type_size]
        weight_main = shuffle_weight_int5(weight_main)
        weight_scale = weight[:, 0:2]
        weight_bias = weight[:, 2:4]
        weight_scale = numpy.frombuffer(weight_scale.tobytes(), numpy.float16).reshape((block_number, 1))
        weight_bias = numpy.frombuffer(weight_bias.tobytes(), numpy.float16).reshape((block_number, 1))
        weight_scale = numpy.concatenate((weight_bias, weight_scale), axis=1).astype(numpy.float32)
        quant_bit = 5
        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, True, mnn_weight_file, ic, oc, weight_main, weight_scale, mnn_weight_offset)
    elif weight.tensor_type == constants.GGMLQuantizationType.Q6_K:
        block_size, type_size = constants.GGML_QUANT_SIZES[weight.tensor_type]
        block_number = oc * ic // block_size
        q_raw, weight_scale, block_size, bits = extract_tensor_as_int8(weight)
        weight_main = repack_low_bits(q_raw, 6, 256)
        quant_bit = 6
        conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, False, mnn_weight_file, ic, oc, weight_main, weight_scale, mnn_weight_offset)

    else:
        print('Not support type: ',  weight.tensor_type)
        print(weight.data.shape, ic, oc)
        assert(False)
    return mnn_weight_offset, conv, tie_embedding, block_size, quant_bit, header_len

def convert(args):
    gguf = args.gguf
    mnn_dir = args.mnn_dir
    src_json = os.path.join(mnn_dir, "llm.mnn.json")
    dst_json = os.path.join(mnn_dir, "llm.mnn_new.json")

    mnn, opmap, convs, _, __ = load_mnn(src_json)
    llm_config = {}
    with open(os.path.join(mnn_dir, "llm_config.json")) as f:
        llm_config = json.load(f)

    reader = gguf_reader.GGUFReader(gguf)
    if args.load_token:
        write_token_file(os.path.join(mnn_dir, "tokenizer.txt"), load_token(reader))
    arch = reader.fields['general.architecture'].parts[4].tobytes().decode('utf-8')
    print("Arch:", arch)
    tensormap = {}
    for t in reader.tensors:
        tensormap[t.name] = t

    mnn_weight_file = open(os.path.join(mnn_dir, "llm.mnn.weight"), "wb")
    mnn_weight_offset = 0
    if 'tie_embeddings' in llm_config:
        del llm_config['tie_embeddings']
    for name in opmap:
        op = opmap[name]
        print('Load layernorm: ', name)
        if op['type'] == 'LayerNorm':
            weight_tensor = tensormap[name+'.weight']
            layernorm = op['main']
            layernorm['gamma'] = weight_tensor.data.tolist()
            if name+'.bias' in tensormap:
                layernorm['beta'] = tensormap[name+'.bias'].data.tolist()
            else:
                layernorm['beta'] = [0.0] * len(layernorm['gamma'])
            continue
    for op in convs:
        conv = op['main']
        name = op['name']
        if 'quanParameter' in conv:
            del conv['quanParameter']
        weight_name = name+'.weight'
        weight = None
        tie_embedding = False
        ichannel = conv['common']['inputCount']
        ochannel = conv['common']['outputCount']
        if name == 'output':
            print('hidden size: ', ichannel)
            llm_config['hidden_size'] = ichannel
        if weight_name in tensormap:
            weight = tensormap[weight_name]
        elif name == 'output':
            weight = tensormap['token_embd.weight']
            tie_embedding = True
        else:
            print("Error: Can't find weight for " + name)
            assert(False)
        print('Load Convolution: ', name, ", weight type: ", weight.tensor_type)
        if weight.shape[0] != ichannel or weight.shape[1] != ochannel:
            print(name, ", weight not match: ", ichannel, ", ", ochannel, " : ", weight.shape, ", reset to ", weight.shape)
            ichannel = int(weight.shape[0])
            ochannel = int(weight.shape[1])
            conv['common']['inputCount'] = ichannel
            conv['common']['outputCount'] = ochannel
            # Change post reshape for convolution
            outputIndex = op['outputIndexes'][0]
            for subop in mnn["oplists"]:
                if 'inputIndexes' not in subop:
                    continue
                if subop['inputIndexes'][0] == outputIndex and subop['type'] == 'ConvertTensor':
                    outputIndex = subop['outputIndexes'][0]
                    break
            for subop in mnn["oplists"]:
                if 'inputIndexes' not in subop:
                    continue
                if subop['inputIndexes'][0] == outputIndex and subop['type'] == 'Reshape':
                    subop['main']['dims'][2] = ochannel
                    break
        mnn_weight_offset, conv_new, can_tie_embedding, block_size, quant_bit, header_len = write_external_weight(weight, mnn_weight_file, mnn_weight_offset)
        if not can_tie_embedding:
            tie_embedding = False
        conv['quanParameter'] = conv_new['quanParameter']
        conv['external'] = conv_new['external']

        bias = None
        bias_name = name + '.bias'
        if bias_name in tensormap:
            if tensormap[bias_name].tensor_type > 1:
                print('Error: Bias is quant: ', tensormap[bias_name].tensor_type)
                assert(False)
            bias = tensormap[bias_name].data.astype(numpy.float32)
        else:
            bias = numpy.zeros(ochannel).astype(numpy.float32)
        mnn_weight_offset += mnn_weight_file.write(bias.tobytes())
        if tie_embedding:
            external = conv['external']
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, quant_bit, 32]
    embedding_file = os.path.join(mnn_dir, "embeddings_bf16.bin")

    embeding_in_weight = True
    if 'tie_embeddings' not in llm_config:
        # Need write embedding
        weight = tensormap['token_embd.weight']
        print("Embedding type: ", weight.tensor_type)
        if weight.tensor_type <= 1:
            embeding_in_weight = False
            print("Write ", embedding_file)
            weight = weight.data.astype(numpy.float32)
            weight = numpy.frombuffer(weight.tobytes(), numpy.uint32) >> 16
            weight = weight.astype(numpy.uint16)
            with open(embedding_file, 'wb') as f:
                f.write(weight.tobytes())
        elif weight.tensor_type == constants.GGMLQuantizationType.Q8_0 or weight.tensor_type == constants.GGMLQuantizationType.Q4_0 or weight.tensor_type == constants.GGMLQuantizationType.Q4_1:
            mnn_weight_offset, conv, can_tie_embedding, block_size, quant_bit, header_len = write_external_weight(weight, mnn_weight_file, mnn_weight_offset)
            external = conv['external']
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, quant_bit, block_size]
        elif weight.tensor_type == constants.GGMLQuantizationType.Q6_K or weight.tensor_type == constants.GGMLQuantizationType.Q5_0:
            q_raw, weight_scale, block_size, bits = extract_tensor_as_int8(weight)
            # embeding_in_weight = False
            ic = int(weight.shape[0])
            oc = int(weight.shape[1])
            offset = (1 << (bits - 1))
            q_raw = repack_low_bits(q_raw, 8, q_raw.shape[1])
            q_raw = q_raw + (128-offset)
            quant_bit = 8
            conv, header_len, mnn_weight_offset = write_quant_parameters(quant_bit, False, mnn_weight_file, ic, oc, q_raw, weight_scale, mnn_weight_offset)
            external = conv['external']
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, quant_bit, block_size]
        else:
            assert(False)

    if embeding_in_weight:
        if os.path.exists(embedding_file):
            os.remove(embedding_file)

    mnn_weight_file.close()
    with open(dst_json, 'w') as f:
        f.write(json.dumps(mnn, indent=4))
    with open(os.path.join(mnn_dir, "llm_config.json"), 'w') as f:
        f.write(json.dumps(llm_config, indent=4))

    convert_args = [
        '',
        '-f',
        'JSON',
        '--modelFile',
        dst_json,
        '--MNNModel',
        os.path.join(mnn_dir, 'llm.mnn'),
    ]

    print(convert_args)
    from MNN.tools import mnnconvert
    mnnconvert.convert(convert_args)
    os.remove(dst_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gguf2mnn', formatter_class=argparse.RawTextHelpFormatter)    
    parser.add_argument('--gguf', type=str, required=True,help='src gguf model')
    parser.add_argument('--mnn_dir', type=str, required=True,help='mnn llm dir')
    parser.add_argument('--load_token', type=bool, default = False, help='Override tokenizer.txt from gguf')
    args = parser.parse_args()
    import time
    sta = time.time()
    convert(args)
    fin = time.time()
    print("Cost time ", fin - sta, " s")
