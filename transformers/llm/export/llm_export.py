import os
import base64
import glob
import json
import shutil
import argparse
import torch
import numpy as np
from onnxslim import slim
import onnxruntime as ort
import sentencepiece as spm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
try:
    import _tools as MNNTools
except:
    MNNTools = None

def onnx2mnn(onnx_path, mnn_dir, quant_bit = 4, asymmetric = True, external_data = False, bizCode : str= None):
    model_name, model_extension = os.path.splitext(os.path.basename(onnx_path))
    if model_extension != '.onnx':
        return
    mnn_name = model_name + '.mnn'
    mnn_path = os.path.join(mnn_dir, mnn_name)
    convert_args = [
        '',
        '-f',
        'ONNX',
        '--modelFile',
        str(onnx_path),
        '--MNNModel',
        str(mnn_path),
        '--weightQuantBits',
        str(quant_bit),
    ]
    if asymmetric:
        convert_args.append("--weightQuantAsymmetric")
    if external_data:
        convert_args.append("--saveExternalData")
    if bizCode is not None:
        convert_args.append("--bizCode")
        convert_args.append(str(bizCode))
    MNNTools.mnnconvert(convert_args)

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, using_bf16: bool = False):
        super().__init__()
        self.bf16 = using_bf16
        self.embed_dim = embed.weight.shape[-1]
        if using_bf16:
            # using bf16 embedding weight
            self.embed = embed.bfloat16()
        else:
            self.embed = embed

    def forward(self, input_ids):
        res = self.embed(input_ids)
        if self.bf16:
            res = res.float()
        return res.view(-1, 1, self.embed_dim)

class Lm(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        # token = torch.argmax(m_logits)
        return m_logits

class LLM(torch.nn.Module):
    '''
    Base class for all llm model. Inherits from [`torch.nn.Module`].
    '''

    def __init__(self, args):
        super().__init__()
        self.quant_bit = 4
        self.asymmetric = True
        self.onnx_path = args.onnx_path
        self.mnn_path = args.mnn_path
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
        if not os.path.exists(self.mnn_path):
            os.makedirs(self.mnn_path)
        self.export_mnn = args.export_mnn
        self.export_verbose = args.export_verbose
        self.export_test = args.export_test
        # default is False, just set True when using below command:
        # `python llm_export ../path --export --embed_bin` to export single model without embedding
        self.without_embed = False
        self.embed_bin = True
        self.embed_bf16 = args.embed_bf16
        self.skip_slim = args.skip_slim
        tokenizer_model = os.path.join(args.path, 'tokenizer.model')
        ice_text_model = os.path.join(args.path, 'ice_text.model')
        try:
            if os.path.exists(tokenizer_model):
                self.sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                self.sp_model = spm.SentencePieceProcessor(ice_text_model)
            else:
                self.sp_model = None
        except:
            self.sp_model = None
        merge_file = os.path.join(args.path, 'merges.txt')
        if os.path.exists(merge_file):
            self.merge_txt = merge_file
        else:
            self.merge_txt = None
        self.stop_ids = []
        self.max_length = 1024
        self.hidden_size = 4096
        self.visual = None # defualt is not visual
        self.lora_path = args.lora_path
        self.load_hf(args.path)
        self.load_model()
        self.llm_config = {
            'hidden_size' : self.hidden_size,
            'layer_nums' : self.block_nums,
            'attention_mask': self.attention_mask_type,
            'key_value_shape': self.past_kv_shape[1:],
            "prompt_template": self.build_prompt('%s'),
            'is_visual': False
        }

    def load_hf(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
        except:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        self.config = self.model.config
        if self.lora_path is not None:
            adapter = PeftModel.from_pretrained(self.model, model_id=self.lora_path)
            self.model = adapter.merge_and_unload(progressbar=True)

    def load_model(self):
        raise NotImplementedError

    def get_attention_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def get_position_ids(self) -> torch.Tensor:
        raise NotImplementedError

    def export_vocab(self):
        raise NotImplementedError

    def visual_embed(self, input_ids):
        raise NotImplementedError

    def __embedding(self, input_ids):
        if self.visual is not None and self.token_len == 0:
            input_embeds = self.visual_embed(input_ids)
        else:
            input_embeds = self.embed(input_ids)
        return input_embeds

    def __decode(self, hidden_states, attention_mask, position_ids, past_key_values):
        presents = []
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            presents.append(kv)
        logits = self.lm(hidden_states).reshape(-1)
        presents = torch.stack(presents)
        self.seq_len += 1
        self.token_len += 1
        return logits, presents

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        if self.without_embed:
            return self.__decode(input_ids, attention_mask, position_ids, past_key_values)
        return self.__decode(self.__embedding(input_ids), attention_mask, position_ids, past_key_values)

    # some test functions
    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def response(self, query):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.numel()
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [None for i in range(self.block_nums)]
        token_id = input_ids
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            logits, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            token_id = torch.argmax(logits)
            if token_id in self.stop_ids:
                print("", end='\n')
                break
            word = self.id_to_str(token_id)
            print(word, end="", flush=True)

    # some export functions
    def assert_equal(self, torch_outs, onnx_outs):
        if type(torch_outs) not in (list, tuple):
            torch_outs = (torch_outs, )
            onnx_outs = (onnx_outs, )
        same = True
        for orig, onnx in zip(torch_outs, onnx_outs):
            orig = orig.detach().numpy()
            if not np.allclose(orig, onnx, rtol=1e-3, atol=1e-3):
                print('Error: onnx outputs dont match original. [shape = {}] onnx: {}, original: {}'.format(onnx.shape, onnx, orig))
                same = False
                break
        if same:
            print('onnx test SUCCESS')

    def export_lm(self):
        model = self.lm
        hidden_states = torch.randn(1, self.hidden_size)
        onnx_model = f'./{self.onnx_path}/lm.onnx'
        torch.onnx.export(model, (hidden_states),
                        onnx_model,
                        verbose=self.export_verbose,
                        input_names=['hidden_states'],
                        output_names=['logits'],
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test lm
        if self.export_test:
            original_outs = model(hidden_states)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'hidden_states' : hidden_states.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)
        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path, self.quant_bit, self.asymmetric)

    def export_visual(self):
        if self.visual is None:
            return
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        model = self.visual
        onnx_model = f'./{self.onnx_path}/visual.onnx'
        torch.onnx.export(model, (input_images),
                        onnx_model,
                        verbose=self.export_verbose,
                        input_names=['input_images'],
                        output_names=['image_embeds'],
                        dynamic_axes={"input_images": {
                            0: "size"
                        }},
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test
        if self.export_test:
            original_outs = model(input_images)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_images' : input_images.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)[0]
            self.assert_equal(original_outs, onnx_outs)
        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path)

    def export_embed(self):
        model = self.embed
        if self.embed_bin:
            import ctypes
            tensor_data = model.embed.weight.data
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            with open(f'./{self.onnx_path}/embeddings_bf16.bin', 'wb') as f:
                f.write(buffer)
            return
        input_ids = torch.arange(3, dtype=torch.long)
        onnx_model = f'./{self.onnx_path}/embedding.onnx'
        torch.onnx.export(model, (input_ids),
                        onnx_model,
                        verbose=self.export_verbose,
                        input_names=['input_ids'],
                        output_names=['inputs_embeds'],
                        dynamic_axes={"input_ids": {
                            0: "length"
                        }},
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test
        if self.export_test:
            original_outs = model(input_ids)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)
        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path)

    def export_block(self, block_id: int):
        self.seq_len = 3
        self.token_len = 0
        inputs_embeds = torch.randn((self.seq_len, 1, self.hidden_size))
        attention_mask =  self.get_attention_mask()
        position_ids = self.get_position_ids()
        past_key_values = torch.zeros(self.past_kv_shape[1:])
        model = self.blocks[block_id]
        onnx_model = f'./{self.onnx_path}/block_{block_id}.onnx'
        torch.onnx.export(
            model, (inputs_embeds, attention_mask, position_ids, past_key_values),
            onnx_model,
            verbose=self.export_verbose,
            input_names=[
                'inputs_embeds', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['hidden_states', 'presents'],
            dynamic_axes=self.block_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        if self.export_test:
            original_outs = model(inputs_embeds, attention_mask, position_ids, past_key_values)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'inputs_embeds' : inputs_embeds.detach().numpy(),
                'attention_mask' : attention_mask.numpy(),
                'position_ids' : position_ids.numpy(),
                'past_key_values' : past_key_values.numpy()
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)
        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path, self.quant_bit, self.asymmetric)

    def export_blocks(self):
        for i in range(self.block_nums):
            self.export_block(i)

    def export_config(self, is_single = True):
        self.llm_config['is_single'] = is_single
        with open(f'./{self.onnx_path}/llm_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.llm_config, f, ensure_ascii=False, indent=4)

    def export(self):
        model = self
        self.seq_len = 3
        self.token_len = 0
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  self.get_attention_mask()
        position_ids = self.get_position_ids()
        past_key_values = torch.zeros(self.past_kv_shape)
        onnx_model = f'./{self.onnx_path}/llm.onnx'
        if self.embed_bin:
            self.without_embed = True
            input_ids = self.__embedding(input_ids)
        print('export start ...')
        torch.onnx.export(
            model, (input_ids, attention_mask, position_ids, past_key_values),
            onnx_model,
            verbose=self.export_verbose,
            input_names=[
                'input_ids', 'attention_mask', 'position_ids', 'past_key_values'
            ],
            output_names=['logits', 'presents'],
            dynamic_axes=self.model_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        print('export done!')
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
            for file_path in glob.glob(f'./{self.onnx_path}/onnx__*'):
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass
            for file_path in glob.glob(f'./{self.onnx_path}/model.*'):
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass
        if self.export_test:
            # test
            original_outs = model(input_ids, attention_mask, position_ids, past_key_values)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.detach().numpy(),
                'attention_mask' : attention_mask.numpy(),
                'position_ids' : position_ids.numpy(),
                'past_key_values' : past_key_values.numpy()
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)
        if self.export_mnn:
            # single model is > 2G, using external_data
            onnx2mnn(onnx_model, self.mnn_path, self.quant_bit, self.asymmetric, True)
        if self.without_embed:
            self.without_embed = False

    def export_tokenizer(self):
        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3
        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')
        def write_header(fp, type, speicals, prefix = []):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        file_path = os.path.join(self.onnx_path, "tokenizer.txt")
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)
        vocab_list = []
        prefix_list = []
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()
        if self.sp_model is not None:
            # senetencepiece
            print('# senetencepiece tokenier')
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            for i in range(self.sp_model.GetPieceSize()):
                token = self.sp_model.IdToPiece(i)
                score = self.sp_model.GetScore(i)
                type = NORMAL
                if self.sp_model.IsUnknown(i):
                    type = UNKNOWN
                elif self.sp_model.IsControl(i):
                    type = CONTROL
                elif self.sp_model.IsUnused(i):
                    type = UNUSED
                elif self.sp_model.IsByte(i):
                    type = BYTE
                if self.model_name == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')
                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {type}\n')
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            print('# tiktoken tokenier')
            # tikton
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif self.merge_txt is not None:
            # huggingface tokenizer
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]
            # load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k
            # load merge
            with open(self.merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)
            # write to tokenizer.txt
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            print('# other tiktoken tokenier')
            # other tikton
            def unicode_to_byte(u: int):
                if u >= 256 and u <= 288:
                    return u - 256
                if u >= 289 and u <= 322:
                    return u - 162
                if u == 323:
                    return 173
                if u == 65372: # |
                    return 124
                if u == 9601:  # _
                    return 95
                return u
            vocab = self.tokenizer.get_vocab()
            vocab_list = ['<unk>' for i in range(len(vocab))]
            for k, v in vocab.items():
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k]).decode('utf-8', errors='ignore')
                except:
                    vocab_list[int(v)] = k
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list)
                fp.write(f'{len(vocab_list)}\n')
                for v in vocab_list:
                    line = base64.b64encode(v.encode('utf-8')).decode("utf8") + "\n"
                    fp.write(line)

# chatglm
class GLMBlock(torch.nn.Module):
    def __init__(self, block, block_id, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.hidden_size = 4096
        self.final_layernorm = final_layernorm

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        hidden_states, presents = self.block(hidden_states,
                                             position_ids,
                                             attention_mask,
                                             self.block_id,
                                             past_kv,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Chatglm_6b(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'glm'
        self.model_name = 'Chatglm_6b'
        super().__init__(args)

    def load_model(self):
        transformer = self.model.transformer
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.word_embeddings
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.final_layernorm
        # some wrapper
        self.stop_ids.append(self.tokenizer._convert_token_to_id(self.tokenizer.eos_token))
        self.block_nums = len(self.blocks_)
        self.lm = Lm(self.lm_)
        # chatglm embedding and lm using same param, copy embedding when using bf16
        if self.embed_bf16:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self.embed_bf16)
        else:
            self.embed = Embedding(self.embed_, self.embed_bf16)
        self.blocks = [GLMBlock(self.blocks_[i], i, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [28, 2, 0, 1, 32, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 2: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 2: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1]).bool().reshape([1, 1, 1, 1])
        attention_mask = torch.zeros([self.seq_len, self.seq_len], dtype=torch.bool)
        for i in range(self.seq_len - 1):
            attention_mask[i][-1] = True
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])
        return attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([self.context_len, self.token_len + 1]).reshape([1, 2, 1])
        position_ids_0 = torch.arange(self.seq_len, dtype=torch.long)
        position_ids_1 = torch.zeros(self.seq_len, dtype=torch.long)
        position_ids_0[-1] = position_ids_0[-2]
        position_ids_1[-1] = 1
        position_ids = torch.stack([position_ids_0, position_ids_1]).view(1, 2, -1)
        return position_ids

    def build_prompt(self, query):
        return f'{query}[gMASK]<sop>'

# chatglm2
class GLM2Block(torch.nn.Module):
    def __init__(self, block, block_id, config, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm
        self.config = config
        self.hidden_size = 4096

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        rope_ratio = self.config.rope_ratio
        base = 10000 * rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).unsqueeze(0).contiguous()
        hidden_states, presents = self.block(hidden_states,
                                            attention_mask,
                                            kv_cache=past_kv,
                                            rotary_pos_emb=rotary_pos_emb)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Chatglm2_6b(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'glm2'
        super().__init__(args)
        self.model_name = 'Chatglm2_6b'
        if 'codegeex2-6b' in args.path:
            self.model_name = 'Codegeex2_6b'

    def load_model(self):
        transformer = self.model.transformer
        self.lm_ = transformer.output_layer
        self.embed_ = transformer.embedding.word_embeddings
        self.blocks_ = transformer.encoder.layers
        self.final_layernorm_ = transformer.encoder.final_layernorm
        # some wrapper
        if self.tokenizer.eos_token_id is None:
            # codegeex2-6b
            self.stop_ids.append(self.tokenizer.tokenizer.eos_id)
        else:
            self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.config, 'eos_token_id'):
            if type(self.config.eos_token_id) is list:
                for eos_id in self.config.eos_token_id:
                    self.stop_ids.append(eos_id)
            elif type(self.config.eos_token_id) is int:
                self.stop_ids.append(self.config.eos_token_id)
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [GLM2Block(self.blocks_[i], i, self.config, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [28, 2, 0, 1, 2, 128]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }
        num_layers = self.config.num_layers
        if num_layers > 28:
            self.past_kv_shape = [num_layers, 2, 1, 2, 0, 128]
            self.block_dynamic_axes = {
                "inputs_embeds" : { 0: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 3: "history_len" }
            }
            self.model_dynamic_axes = {
                "input_ids" : { 0: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 0: "seq_len" },
                "past_key_values" : { 4: "history_len" }
            }

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, 1]).bool()
        attention_mask = ~torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]).bool())
        return attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([self.token_len], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long)

# chatglm3
class Chatglm3_6b(Chatglm2_6b):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'Chatglm3_6b'

    def build_prompt(self, query):
        return f'<|user|>\n{query}\n<|assistant|>\n'

# qwen
class QWENBlock(torch.nn.Module):
    def __init__(self, name, block, block_id, hidden_size, final_layernorm = None):
        super().__init__()
        self.name = name
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(0)
        if self.name != 'Qwen-7B':
            rotary_pos_emb = torch.stack([torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb)])
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states, presents = self.block(hidden_states=hidden_states,
                                             layer_past=past_kv,
                                             attention_mask=attention_mask,
                                             rotary_pos_emb=rotary_pos_emb,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class QWEN18Block(torch.nn.Module):
    def __init__(self, block, block_id, hidden_size, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1).unsqueeze(1).unsqueeze(0)
        rotary_pos_emb = torch.stack([torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb)])
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states, presents = self.block(hidden_states,
                                             rotary_pos_emb,
                                             past_kv,
                                             attention_mask,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Qwen_Chat(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'int'
        super().__init__(args)
        if 'VL' in self.model_name:
            self.llm_config['is_visual'] = True
            self.llm_config['attention_mask'] = 'float'
            self.llm_config['img_size'] = 448
            self.llm_config['imgpad_len'] = 256
            self.llm_config['img_start'] = self.tokenizer.img_start_id
            self.llm_config['img_end'] = self.tokenizer.img_end_id
            self.llm_config['img_pad'] = self.tokenizer.img_pad_id


    def load_model(self):
        # Qwen models
        self.model_name = 'Qwen-7B'
        if '1_8' in model_path:
            self.model_name = 'Qwen-1_8b'
        if 'VL' in model_path:
            self.model_name = 'Qwen-VL'
        transformer = self.model.transformer
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.wte
        self.blocks_ = transformer.h
        self.final_layernorm_ = transformer.ln_f
        if hasattr(transformer, 'visual'):
            self.visual = transformer.visual
            self.image_start_id = transformer.config.visual['image_start_id']
            self.image_size = transformer.config.visual['image_size']
        # some wrapper
        self.stop_ids.append(self.tokenizer.im_end_id)
        self.block_nums = len(self.blocks_)
        self.hidden_size = transformer.embed_dim
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [QWENBlock(self.model_name, self.blocks_[i], i, self.hidden_size, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        if self.block_nums == 32:
            # qwen-7b, qwen-vl
            self.past_kv_shape = [32, 2, 1, 0, 32, 128]
        elif self.block_nums == 24:
            # qwen-1.8b
            self.past_kv_shape = [24, 2, 1, 0, 16, 128]
        # some config for export
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 3: "history_len" }
        }

    def build_prompt(self, query):
        return f'\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    def get_attention_mask(self) -> torch.Tensor:
        if self.model_name == 'Qwen-VL':
            if self.token_len:
                return torch.zeros([1, 1, 1, 1], dtype=torch.float32)
            return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
        if self.token_len:
            return torch.ones([1, 1, 1, 1]).bool()
        return torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]).bool())

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([self.seq_len - 1], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long)

    def visual_embed(self, input_ids):
        if not torch.any(input_ids == self.image_start_id):
            return self.embed(input_ids)
        bos_pos = torch.where(input_ids == self.image_start_id)
        eos_pos = torch.where(input_ids == self.image_start_id + 1)
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[ : image.index(self.image_start_id + 2)]
            images.append(bytes(image).decode('utf-8'))
        images = self.visual.encode(images)
        hidden_states = self.embed(input_ids).view(1, -1, self.hidden_size)
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1 : b] = images[idx]
        return hidden_states.view(-1, 1, self.hidden_size)

class QWEN2Block(torch.nn.Module):
    def __init__(self, name, block, block_id, config, final_layernorm = None):
        super().__init__()
        self.name = name
        self.block = block
        self.block_id = block_id
        self.final_layernorm = final_layernorm
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(0)
        rotary_pos_emb = torch.stack([torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb)])
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states, presents = self.block(hidden_states=hidden_states,
                                             attention_mask=attention_mask,
                                             past_key_value=past_kv,
                                             rotary_pos_emb=rotary_pos_emb,
                                             use_cache=True)

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        # print('###', presents.shape)
        return hidden_states, presents

class Qwen2_Chat(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'float'
        super().__init__(args)

    def load_model(self):
        # Qwen2 models
        self.model_name = 'Qwen2'
        transformer = self.model.model
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.embed_tokens
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.norm
        # some wrapper
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.model, 'generation_config'):
            for id in self.model.generation_config.eos_token_id:
                self.stop_ids.append(id)
        self.block_nums = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.kv_heads = self.config.num_key_value_heads
        self.rope_theta = self.config.rope_theta
        self.head_dim = self.hidden_size // self.num_heads
        if self.embed_.weight is self.lm_.weight:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self.embed_bf16)
        else:
            self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.past_kv_shape = [self.block_nums, 2, 1, 0, self.kv_heads, self.head_dim]
        self.blocks = [QWEN2Block(self.model_name, self.blocks_[i], i, self.config, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        # some config for export
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }

    def build_prompt(self, query):
        return f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min


    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([[self.seq_len - 1]], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def visual_embed(self, input_ids):
        if not torch.any(input_ids == self.image_start_id):
            return self.embed(input_ids)
        bos_pos = torch.where(input_ids == self.image_start_id)
        eos_pos = torch.where(input_ids == self.image_start_id + 1)
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[ : image.index(self.image_start_id + 2)]
            images.append(bytes(image).decode('utf-8'))
        images = self.visual.encode(images)
        hidden_states = self.embed(input_ids).view(1, -1, self.hidden_size)
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1 : b] = images[idx]
        return hidden_states.view(-1, 1, self.hidden_size)

# llama2
class LLAMA2Block(torch.nn.Module):
    def __init__(self, block, block_id, hidden_size, head_dim, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.head_dim = head_dim
        self.final_layernorm = final_layernorm
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.cat((idx_theta, idx_theta), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(0)
        rotary_pos_emb = torch.stack([torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb)])
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        position_ids = position_ids.view(1, -1)
        hidden_states, presents = self.block(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             past_kv,
                                             rotary_pos_emb=rotary_pos_emb,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        if isinstance(presents, tuple):
            presents = torch.stack(presents)
        return hidden_states, presents

class Llama2_7b_Chat(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'float'
        self.model_name = 'Llama2_7b'
        if 'Baichuan2' in args.path:
            self.model_name = 'Baichuan2_7B'
        if 'internlm' in args.path:
            self.model_name = 'Internlm_7b'
        if 'TinyLlama' in args.path:
            self.model_name = 'TinyLlama'
        if 'Yi' in args.path:
            self.model_name = 'Yi'
        if 'deepseek' in args.path:
            self.model_name = 'deepseek'
        if 'Llama-3' in args.path:
            self.model_name = 'Llama3_8B'
        super().__init__(args)

    def load_model(self):
        self.config = self.model.config
        transformer = self.model.model
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.embed_tokens
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.norm
        # some wrapper
        self.hidden_size = self.embed_.weight.shape[-1]
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.model, 'generation_config'):
            self.stop_ids.append(self.model.generation_config.eos_token_id)
        if self.model_name == 'Llama3_8B':
            self.stop_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.block_nums = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        if hasattr(self.config, 'num_key_value_heads'):
            self.num_key_value_heads = self.config.num_key_value_heads
        else:
            self.num_key_value_heads = self.config.num_attention_heads
        self.blocks = [LLAMA2Block(self.blocks_[i], i, self.hidden_size, self.head_dim, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        self.past_kv_shape = [self.block_nums, 2, 1, 0, self.num_key_value_heads, self.head_dim]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 1: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 1: "seq_len" },
            "past_key_values" : { 3: "history_len" }
        }

    def build_prompt(self, query):
        if 'Baichuan2' in self.model_name:
            return f'<reserved_106>{query}<reserved_107>'
        if 'Internlm_7b' in self.model_name:
            return f'<|User|>:{query}<eoh>\n<|Bot|>:'
        if 'TinyLlama' in self.model_name:
            return f'<s><|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n{query}</s>\n<|assistant|>\n'
        if 'Yi' in self.model_name:
            return f'<|im_start|> user\n{query}<|im_end|>\n<|im_start|> assistant\n'
        if 'deepseek' in self.model_name:
            return f'<|begin_of_sentence|>User: {query}\n\nAssistant:'
        if 'Llama3' in self.model_name:
            return f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        return f'<s>[INST]{query}[/INST]'

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, self.seq_len], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([[self.seq_len - 1]], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

# phi-2
class PHI2Block(torch.nn.Module):
    def __init__(self, block, block_id, hidden_size):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        theta = 1.0 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=0).contiguous()
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states, presents = self.block(hidden_states,
                                             past_kv,
                                             rotary_pos_emb=rotary_pos_emb,
                                             causal_mask=attention_mask
                                             )
        if self.block_id == 31:
            hidden_states = hidden_states[:, -1, :]
        return hidden_states, presents

class phi_2(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'glm'
        super().__init__(args)
        self.model_name = 'phi-2'
        self.asymmetric = False # TODO: some precision bug when using asymmetric

    def load_model(self):
        transformer = self.model.transformer
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.embd.wte
        self.hidden_size = self.embed_.weight.shape[-1]
        self.blocks_ = transformer.h
        # self.final_layernorm_ = transformer.final_layernorm
        # some wrapper
        self.stop_ids.append(self.tokenizer.eos_token_id)
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_, self.embed_bf16)
        self.lm = Lm(self.lm_)
        self.blocks = [PHI2Block(self.blocks_[i], i, self.hidden_size) for i in range(self.block_nums)]
        # some config for export
        self.past_kv_shape = [len(self.blocks), 1, 0, 2, 32, 80]
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 1: "history_len" }
        }
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 0: "seq_len" },
            "past_key_values" : { 2: "history_len" }
        }

    def build_prompt(self, query):
            return f'Instruct: {query}\nOutput:'

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, 1]).bool()
        attention_mask = ~torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]).bool())
        return attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.tensor([[self.seq_len - 1]], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

# BGE is Embedding Model based Bert
class BGEBlock(torch.nn.Module):
    def __init__(self, block, block_id, hidden_size):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.block(hidden_states, attention_mask)[0]
        return hidden_states

class bge(LLM):
    def __init__(self, args):
        self.attention_mask_type = 'int'
        self.past_kv_shape = []
        super().__init__(args)
        self.model_name = 'bge-large-zh'

    def forward(self, input_ids, position_ids, attention_mask):
        input_ids = input_ids.view(1, -1)
        token_type_ids = (1 - attention_mask).view(1, -1)
        hidden_states = self.embed(input_ids, token_type_ids, position_ids)[0].unsqueeze(0)
        for i in range(self.block_nums):
            hidden_states = self.blocks[i](hidden_states, attention_mask)
        # hidden_states = self.lm(hidden_states) # sentence_embeddings not need
        sentence_embeddings = hidden_states[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def response(self, query):
        self.eval()
        input_ids = self.tokenizer(query)['input_ids']
        self.seq_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        position_ids = self.get_position_ids()
        attention_mask = self.get_attention_mask()
        res = self.forward(input_ids, position_ids, attention_mask)
        return res

    def load_model(self):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        transformer = self.model.encoder
        self.lm_ = self.model.pooler
        self.embed_ = self.model.embeddings
        self.hidden_size = self.embed_.word_embeddings.weight.shape[-1]
        self.blocks_ = transformer.layer
        # some wrapper
        self.stop_ids = []
        self.block_nums = len(self.blocks_)
        self.embed = self.embed_
        self.lm = self.lm_
        self.blocks = [BGEBlock(self.blocks_[i], i, self.hidden_size) for i in range(self.block_nums)]
        # some config for export
        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "position_ids" : { 1: "seq_len" },
            "attention_mask" : { 3: "seq_len" }
        }

    def export(self):
        model = self.eval()
        self.seq_len = 3
        input_ids = torch.arange(3, dtype=torch.long)
        position_ids = self.get_position_ids()
        attention_mask = self.get_attention_mask()
        onnx_model = f'./{self.onnx_path}/bge.onnx'
        torch.onnx.export(
            model, (input_ids, position_ids, attention_mask),
            onnx_model,
            verbose=self.export_verbose,
            input_names=[
                'input_ids',
                'position_ids',
                'attention_mask'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes=self.model_dynamic_axes,
            do_constant_folding=True,
            opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        if self.export_test:
            self.seq_len = 4
            position_ids = self.get_position_ids()
            input_ids = torch.tensor([ 101,  872, 1962,  102 ], dtype=torch.long)
            attention_mask = self.get_attention_mask()
            # test
            original_outs = model(input_ids, position_ids, attention_mask)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.detach().numpy(),
                'position_ids' : position_ids.detach().numpy(),
                'attention_mask' : attention_mask.detach().numpy()
            }
            onnx_outs = ort_session.run(None, inputs)[0]
            self.assert_equal(original_outs, onnx_outs)

        token_str = None
        if False: # save tokenizer in mnn
            self.export_tokenizer()
            token_path = os.path.join(self.onnx_path, "tokenizer.txt")
            token_str = open(token_path, 'rt').read()

        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path, 8, True, bizCode=token_str)

    def build_prompt(self, query):
            return f'[CLS]{query}[SEP]'

    def get_position_ids(self) -> torch.Tensor:
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)

    def get_attention_mask(self) -> torch.Tensor:
        return torch.ones([1, 1, 1, self.seq_len], dtype=torch.long)

class LoraModule(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.onnx_path = args.onnx_path
        self.mnn_path = args.mnn_path
        self.export_mnn = args.export_mnn
        import peft
        lora_weight = peft.load_peft_weights(args.path)
        for k, v in lora_weight.items():
            k = k.replace('.', '/')
            self.register_buffer(k, v.cpu())

    def forward(self, dummpy):
        return self._buffers

    def export(self):
        onnx_model = f'./{self.onnx_path}/lora.onnx'
        torch.onnx.export(self.eval(), torch.tensor([]), onnx_model)
        if self.export_mnn:
            onnx2mnn(onnx_model, self.mnn_path)


if __name__ == '__main__':
    llm_models = {
        'chatglm-6b': Chatglm_6b,
        'chatglm2-6b': Chatglm2_6b,
        'codegeex2-6b': Chatglm2_6b,
        'chatglm3-6b': Chatglm3_6b,
        'glm-4-9b-chat': Chatglm3_6b,
        'Qwen-7B-Chat': Qwen_Chat,
        'Qwen-1_8B-Chat': Qwen_Chat,
        'Qwen-1_8B': Qwen_Chat,
        'Qwen-VL-Chat': Qwen_Chat,
        'Qwen1_5-0_5B-Chat': Qwen2_Chat,
        'Qwen1_5-1_8B-Chat': Qwen2_Chat,
        'Qwen1_5-4B-Chat': Qwen2_Chat,
        'Qwen1_5-7B-Chat': Qwen2_Chat,
        'Qwen2-0_5B-Instruct': Qwen2_Chat,
        'Qwen2-1_5B-Instruct': Qwen2_Chat,
        'Qwen2-7B-Instruct': Qwen2_Chat,
        'Baichuan2-7B-Chat': Llama2_7b_Chat,
        'Llama-2-7b-chat-ms': Llama2_7b_Chat,
        'Llama-3-8B-Instruct': Llama2_7b_Chat,
        'internlm-chat-7b': Llama2_7b_Chat,
        'TinyLlama-1_1B-Chat': Llama2_7b_Chat,
        'Yi-6B-Chat': Llama2_7b_Chat,
        'deepseek-llm-7b-chat': Llama2_7b_Chat,
        'MiniCPM-1.2b': Llama2_7b_Chat,
        'MiniCPM-2.4b': Llama2_7b_Chat,
        'phi-2': phi_2,
        'bge-large-zh': bge,
        'lora': LoraModule
    }
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, default='THUDM/chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, choices=llm_models.keys(), default=None,
                        help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--lora_path', type=str, default=None, help='lora path, defaut is `None` mean not apply lora.')
    parser.add_argument('--onnx_path', type=str, default='./onnx', help='export onnx model path, defaut is `./onnx`.')
    parser.add_argument('--mnn_path', type=str, default='./mnn', help='export mnn model path, defaut is `./mnn`.')
    parser.add_argument('--export_mnn', action='store_true', default=False, help='Whether or not to export mnn model after onnx.')
    parser.add_argument('--export_verbose', action='store_true', default=False, help='Whether or not to export onnx with verbose.')
    parser.add_argument('--export_test', action='store_true', help='Whether or not to export onnx with test using onnxruntime.')
    parser.add_argument('--test', type=str, help='test model inference with query `TEST`.')
    parser.add_argument('--export', action='store_true', help='export model to an `onnx` model.')
    parser.add_argument('--export_split', action='store_true',
                        help='export model split to some `onnx` models:'
                        '\n\t- embedding model.'
                        '\n\t- block models.'
                        '\n\t- lm_head model.'
                        )
    parser.add_argument('--export_visual', action='store_true', help='export llm visual model to an `onnx` model.')
    parser.add_argument('--export_lm', action='store_true', help='export llm lm_head to an `onnx` model.')
    parser.add_argument('--export_block', type=int, help='export llm block [id] to an `onnx` model.')
    parser.add_argument('--export_blocks', action='store_true', help='export llm all blocks to `onnx` models.')
    parser.add_argument('--skip_slim', action='store_true', help='Whether or not to skip onnx-slim.')

    # No use now, add invoid of call error
    parser.add_argument('--export_token', action='store_true', help='export llm tokenizer to a txt file.')
    parser.add_argument('--export_embed', action='store_true', help='export llm embedding to an `onnx` model.')
    parser.add_argument('--embed_bf16', default=True, action='store_true', help='using `bfloat16` replace `float32` in embedding.')
    parser.add_argument('--embed_bin', action='store_true', help='export embedding weight as bin file with dtype `bfloat16`')

    args = parser.parse_args()
    model_path = args.path
    model_type = args.type
    # not sepcify model type, using path
    if model_type is None:
        for model in llm_models:
            if model in model_path:
                model_type = model
    if model_type is None:
        raise RuntimeError('Please specify model type.')

    # copy modeling py file to pretrain model for export
    for file in glob.glob(f'./llm_models/{model_type}/*'):
        shutil.copy2(file, model_path)

    llm_exporter = llm_models[model_type](args)

    # some actions
    if args.test is not None:
        llm_exporter.response(args.test)

    if args.export or args.export_split:
        llm_exporter.export_config(args.export)

    if args.export:
        llm_exporter.export()

    llm_exporter.export_tokenizer()

    llm_exporter.export_embed()

    if args.export_visual or args.export_split:
        llm_exporter.export_visual()

    if args.export_lm or args.export_split:
        llm_exporter.export_lm()

    if args.export_blocks or args.export_split:
        llm_exporter.export_blocks()

    if args.export_block is not None:
        llm_exporter.export_block(args.export_block)