from utils.mnn_utils import *
import numpy
import os
import argparse
import safetensors
import utils.torch_utils as torch_utils

class Writer:
    def __init__(self, name):
        self.offset = 0
        self.name = name
        self.file = open(name, 'wb')
    def write(self, bytes):
        new_offset = self.file.write(bytes)
        self.offset += new_offset
        return new_offset
    def close(self):
        self.file.close()

class QuantParameter:
    def __init__(self, bits, block, asymc):
        self.bits = bits
        self.block = block
        self.asymc = asymc

def load_embedding(k, f, writer, quant):
    print("Load embedding: ", k)
    weight = f.get_tensor(k).float().cpu().numpy()
    weight = weight.astype(numpy.float32)
    weight = numpy.frombuffer(weight.tobytes(), numpy.uint32) >> 16
    weight = weight.astype(numpy.uint16)
    writer.write(weight.tobytes())
    return

def load_convolution(op, k, f, writer, quant):
    print("Load ", k, " to ", op['name'])
    conv = op['main']
    ic = conv['common']['inputCount']
    oc = conv['common']['outputCount']
    bias_key = k.split(".weight")[0]+".bias"
    bias_tensor = None
    print(f.get_tensor(k).shape)
    if bias_key in f.keys():
        bias_tensor = f.get_tensor(k.split(".weight")[0]+".bias").float().cpu().numpy().astype(numpy.float32)
    else:
        bias_tensor = numpy.zeros([oc], numpy.float32)
    q_weight, alpha = torch_utils.quant(f.get_tensor(k), quant.bits, quant.block, not quant.asymc, False)
    q_weight = q_weight.cpu().numpy()
    alpha = alpha.cpu().numpy()
    conv_new, header_len, mnn_weight_offset = write_quant_parameters(quant.bits, quant.asymc, writer, ic, oc, q_weight, alpha, writer.offset, False)
    conv['quanParameter'] = conv_new['quanParameter']
    conv['external'] = conv_new['external']
    writer.write(bias_tensor.tobytes())
    return header_len, mnn_weight_offset


def load_layernorm(op, k, f):
    layernorm = op['main']
    tensor = f.get_tensor(k)
    layernorm['gamma'] = tensor.float().cpu().numpy().tolist()
    layernorm['beta'] = [0.0] * len(layernorm['gamma'])
    return

def convert(args):
    model = args.path
    mnn_dir = args.mnn_dir
    src_json = os.path.join(mnn_dir, "llm.mnn.json")
    dst_json = os.path.join(mnn_dir, "llm.mnn_new.json")
    mnn, opmap, convs, blockes, last = load_mnn(src_json)
    output_norm = last.attn_norm
    lm = last.conv[0]
    asym = not args.sym
    conv_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    quan = QuantParameter(args.quant_bit, args.quant_block, asym)
    writer = Writer(os.path.join(mnn_dir, "llm.mnn.weight"))
    embedding_with_output = True
    embedding_file_name = None
    embedding_key = "model.embed_tokens.weight"
    for filename in os.listdir(model):
        if not filename.endswith("safetensors"):
            continue
        with safetensors.safe_open(os.path.join(model, filename), framework="pt", device='cpu') as f:
            for k in f.keys():
                if k.endswith(".bias"):
                    # bias will be load together with weight
                    continue
                if k.startswith("model."):
                    # LLM's tensor
                    if k.find("layers.") >=0:
                        # In Block
                        index = int(k.split("layers.")[1].split(".")[0])
                        block = blockes[index]
                        if k.find("input_layernorm") >= 0:
                            load_layernorm(block.layernorm[0], k, f)
                            continue
                        if k.find("post_attention_layernorm") >= 0:
                            load_layernorm(block.layernorm[1], k, f)
                            continue
                        mlp_index = -1
                        for i in range(len(conv_names)):
                            if k.find(conv_names[i]) >=0:
                                mlp_index = i
                                break
                        assert(mlp_index >= 0)
                        load_convolution(block.conv[mlp_index], k, f, writer, quan)
                    elif k == "model.norm.weight":
                        load_layernorm(output_norm, k, f)
                    elif k == embedding_key:
                        embedding_file_name = os.path.join(model, filename)
                    continue
                elif k == "lm_head.weight":
                    embedding_with_output = False
                    quan_int8 = QuantParameter(args.lm_quant_bit, args.quant_block, asym)
                    load_convolution(lm, k, f, writer, quan_int8)
    llm_config = {}
    with open(os.path.join(mnn_dir, "llm_config.json")) as f:
        llm_config = json.load(f)
    with safetensors.safe_open(embedding_file_name, framework="pt", device='cpu') as f:
        if embedding_with_output:
            lmbit = args.lm_quant_bit
            if lmbit != 4 and lmbit != 8:
                if lmbit < 4:
                    lmbit = 4
                else:
                    lmbit = 8
                print("Don't support quant bit", args.lm_quant_bit, " for tie embedding, turn to quant bit", lmbit)
            quan_int8 = QuantParameter(lmbit, args.quant_block, asym)
            header_len, offset = load_convolution(lm, embedding_key, f, writer, quan_int8)
            external = lm['main']['external']
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, lmbit, args.quant_block]
            embedding_file = os.path.join(mnn_dir, "embeddings_bf16.bin")
            if os.path.exists(embedding_file):
                os.remove(embedding_file)
        else:
            emb_writer = Writer(os.path.join(mnn_dir, "embeddings_bf16.bin"))
            load_embedding(embedding_key, f, emb_writer, None)

    writer.close()

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
    parser = argparse.ArgumentParser(description='saftensors2mnn', formatter_class=argparse.RawTextHelpFormatter)    
    parser.add_argument('--path', type=str, required=True,help='src model')
    parser.add_argument('--mnn_dir', type=str, required=True,help='mnn llm dir')
    parser.add_argument('--quant_bit', type=int, default=4, help='mnn quant bit, 2-8, default is 4.')
    parser.add_argument('--quant_block', type=int, default=128, help='mnn quant block, default is 0 mean channle-wise.')
    parser.add_argument('--lm_quant_bit', type=int, default=None, help='mnn lm_head quant bit, 2-8, default is `quant_bit`.')
    parser.add_argument('--sym', type = bool, default=True, help='Whether or not to using symmetric quant (without zeropoint), defualt is True.')

    args = parser.parse_args()
    if args.lm_quant_bit is None:
        args.lm_quant_bit = args.quant_bit

    import time
    sta = time.time()
    convert(args)
    fin = time.time()
    print("Cost time ", fin - sta, "s")
