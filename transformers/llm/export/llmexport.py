import os
import json
import glob
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import onnx
import torch

from utils.model import LlmModel, EmbeddingModel
from utils.tokenizer import LlmTokenizer
from utils.spinner import spinner_run
from utils.custom_op import FakeLinear
from utils.onnx_rebuilder import OnnxRebuilder
from utils.mnn_converter import MNNConverter
from utils.awq_quantizer import AwqQuantizer
from utils.smooth_quantizer import SmoothQuantizer
from utils.omni_quantizer import OmniQuantizer
from utils.torch_utils import onnx_export

class LlmExporter(torch.nn.Module):
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''
    def __init__(self, args):
        super().__init__()
        self.init_from_args(args)
        self.load_model(args.path)

    def init_from_args(self, args):
        self.args = args
        self.max_new_tokens = 1024
        self.dst_name = 'llm'
        # load config from args
        self.onnx_path = os.path.join(self.args.dst_path, 'onnx')
        if self.args.tokenizer_path is None:
            self.args.tokenizer_path = self.args.path
        if args.lm_quant_bit is None:
            self.args.lm_quant_bit = self.args.quant_bit
        if args.lm_quant_block is None:
            self.args.lm_quant_block = self.args.quant_block
        self.args.tie_word_embeddings = False
        # init export dst dir
        if not os.path.exists(self.args.dst_path):
            os.makedirs(self.args.dst_path)
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)

    @spinner_run(f'load pretrained model ', True)
    def load_model(self, model_path):
        self.model = LlmModel.from_pretrained(model_path, args=self.args)
        self.tokenizer = LlmTokenizer.from_pretrained(
            self.args.tokenizer_path,
            model_type=self.model.config.model_type
        )
        self.model.tokenizer = self.tokenizer
        self.config = self.model.config
        self.model_type = self.config.model_type

        if self.args.awq or self.args.smooth:
            self.model.float()
        if self.args.export is not None:
            # set norm's weight as float for export
            def visit_module(module):
                if not isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                    module.float()
                for name, child in module.named_children():
                    visit_module(child)
            visit_module(self.model)

        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 1: "seq_len" },
        }

        self.llm_config = {
            'model_type': self.config.model_type,
            'hidden_size' : self.config.hidden_size,
            'attention_mask': 'float', # Will be determined by model later
            'attention_type': self.config.attention_type,
            'is_mrope': self.model.rotary.is_mrope
        }
        self.llm_config.update(self.model.get_config())
        if self.config.sliding_window > 0:
            self.llm_config['sliding_window'] = self.config.sliding_window
        if hasattr(self.tokenizer, 'get_chat_template'):
             chat_template = self.tokenizer.get_chat_template()
             if chat_template is not None:
                 self.llm_config['jinja'] = {
                     'chat_template': chat_template
                 }
                 if self.tokenizer.bos_token:
                     self.llm_config['jinja']['bos'] = self.tokenizer.bos_token
                 if self.tokenizer.eos_token:
                     self.llm_config['jinja']['eos'] = self.tokenizer.eos_token

        # tie word embeddings
        self.args.tie_word_embeddings = not self.args.seperate_embed and self.model.lm.lm.weight.equal(self.model.embed.embed.weight)
        # Pass properties from model to exporter
        self.visual = self.model.visual
        self.audio = self.model.audio
        self.talker = self.model.talker
        self.mtp = self.model.mtp
        self.scale_emb = self.model.scale_emb

        return model_path

    @torch.no_grad()
    def response(self, query):
        # self.imitate_quant()
        self.model.decode_buffer = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        prompt = self.tokenizer.apply_chat_template(messages)
        if query not in prompt:
            prompt = query

        # Use model's tokenizer methods for encoding
        if self.model.visual is not None:
            input_ids = self.model.visual.str_to_ids(prompt)
        elif self.model.audio is not None:
            input_ids = self.model.audio.str_to_ids(prompt)
        else:
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids']

        seq_len = input_ids.numel()
        new_tokens = 0

        while new_tokens < self.max_new_tokens:
            attention_mask = self.model.get_attention_mask(seq_len, new_tokens)
            position_ids = self.model.get_position_ids(seq_len, new_tokens, input_ids)
            input_embeds = self.model.embedding(input_ids)
            deepstack_embeds = self.model.visual.deepstacks() if self.model.visual is not None else None

            logits, _, _ = self.model.forward(
                input_ids=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_index=torch.tensor([-1], dtype=torch.int32),
                deepstack_embeds=deepstack_embeds
            )

            token_id = torch.argmax(logits[:,-1,:])
            seq_len += 1
            new_tokens += 1
            if token_id in self.tokenizer.stop_ids:
                print("", end='\n')
                break

            # Use tokenizer's method for decoding
            word = self.tokenizer.id_to_str(token_id)
            print(word, end="", flush=True)
            input_ids = token_id

        if hasattr(self.model, 'talker') and self.model.talker is not None:
            self.model.talker.generate()

    def export_mtp(self):
        if self.mtp is None:
            return
        mtp_onnx = self.mtp.export(self.onnx_path)
        if self.mnn_converter:
            self.mtp.unloaded_ops['/lm/lm_head/Linear'] = self.unloaded_ops['/lm/lm_head/Linear']
            MNNConverter(self, self.mtp.unloaded_ops).export(mtp_onnx)

    def export_eagle(self):
        if self.args.eagle_path is None:
            return
        from utils.eagle import Eagle
        self.eagle = Eagle.get_eagle(self.model_type)(self.args.eagle_path, self.model)
        eagle_onnx, eagle_fc_onnx = self.eagle.export(self.onnx_path)
        if self.mnn_converter:
            MNNConverter(self, None).export(eagle_onnx)
            MNNConverter(self, None).export(eagle_fc_onnx)


    @spinner_run(f'export embedding to ')
    def export_embed(self):
        import ctypes
        from utils.torch_utils import quant as torch_quant

        if hasattr(self.model, 'word_embeddings'):
            # embedding model's embed
            tensor_data = self.model.word_embeddings.weight.data
        else:
            tensor_data = self.model.embed.embed.weight.data

        format_bit = getattr(self.args, 'embed_bit', 16)
        quant_block = getattr(self.args, 'quant_block', 64)
        symmetric = getattr(self.args, 'sym', False)

        if self.args.skip_weight:
            format_name = f'int{format_bit}' if format_bit < 16 else 'bf16'
            embedding_file = f'{self.args.dst_path}/embeddings_{format_name}.bin'
            # Calculate expected size
            if format_bit == 16:
                file_size = tensor_data.numel() * 2
            else:
                oc, ic = tensor_data.shape
                block_size = ic if quant_block == 0 else quant_block
                block_num = ic // block_size
                q_weight_size = (oc * ic * format_bit + 7) // 8
                alpha_size = oc * block_num * (1 if symmetric else 2) * 4
                file_size = q_weight_size + alpha_size
                self.llm_config['tie_embeddings'] = [0, q_weight_size, alpha_size, format_bit, quant_block]

            with open(embedding_file, 'wb') as f:
                if file_size > 0:
                    f.seek(file_size - 1)
                    f.write(b'\0')
            return embedding_file

        if format_bit == 16:
            # BF16 format
            tensor_data = tensor_data.bfloat16()
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            embedding_file = f'{self.args.dst_path}/embeddings_bf16.bin'
            with open(embedding_file, 'wb') as f:
                f.write(buffer)
        elif format_bit in [8, 4]:
            # Quantized formats
            quant_bit = format_bit
            format_name = f'int{format_bit}'
            awq = getattr(self.args, 'awq', False)
            hqq = getattr(self.args, 'hqq', False)

            # Apply quantization
            q_weight, alpha = torch_quant(tensor_data.float(), quant_bit, quant_block, symmetric, awq, hqq)

            # Save quantized weights and scales together in one file
            embedding_file = f'{self.args.dst_path}/embeddings_{format_name}.bin'
            with open(embedding_file, 'wb') as f:
                weight_size = f.write(q_weight.numpy().tobytes())
                alpha_size = f.write(alpha.numpy().tobytes())
            self.llm_config['tie_embeddings'] = [0, weight_size, alpha_size, quant_bit, quant_block]
        else:
            raise ValueError(f"Unsupported embedding bit precision: {format_bit}")

        return embedding_file

    @spinner_run(f'export config to ')
    def export_config(self, mnn_config = False):
        with open(f'{self.args.dst_path}/export_args.json', 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, ensure_ascii=False, indent=4)
        config_json = f'{self.args.dst_path}/llm_config.json'
        with open(config_json, 'w', encoding='utf-8') as f:
            json.dump(self.llm_config, f, ensure_ascii=False, indent=4)
        if not mnn_config:
            return config_json
        with open(f'{self.args.dst_path}/config.json', 'w', encoding='utf-8') as f:
            config = {
                "llm_model": f"{self.dst_name}.mnn",
                "llm_weight": f"{self.dst_name}.mnn.weight",
                "backend_type": "cpu",
                "thread_num": 4,
                "precision": "low",
                "memory": "low",
                # "system_prompt": "You are a helpful assistant.",
                "sampler_type":'penalty',
                "penalty":1.1
            }
            if self.args.embed_bit < 16:
                config['embedding_file'] = f"embeddings_int{self.args.embed_bit}.bin"
            if hasattr(self, 'talker') and self.talker is not None:
                config['system_prompt'] = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                config['talker_max_new_tokens'] = 2048
                config['talker_speaker'] = "Chelsie"
                config['dit_steps'] = 5
                config['dit_solver'] = 1
            if self.model_type == "gemma3":
                config.update({'precision': "normal"})
            if (hasattr(self, 'visual') and self.visual is not None) or (hasattr(self, 'visual') and self.audio is not None):
                config['mllm'] = {
                    'backend_type': "cpu",
                    "thread_num": 4,
                    "precision": "normal",
                    "memory": "low"
                }
            if self.args.eagle_path is not None:
                config['speculative_type'] = 'eagle'
                config['hidden_states'] = True
            json.dump(config, f, ensure_ascii=False, indent=4)
        return config_json

    def imitate_quant(self):
        def quant_dequant(linear, quant_bit = self.args.quant_bit, quant_block = self.args.quant_block):
            weight = linear.weight.data
            oc, ic = weight.shape
            if quant_block == 0:
                block_size = ic
            else:
                block_size = quant_block
            block_num = ic // block_size
            weight = weight.reshape(oc, block_num, block_size)
            max_val = torch.max(weight, axis=-1, keepdims=True).values
            min_val = torch.min(weight, axis=-1, keepdims=True).values
            offset = 1 << (quant_bit - 1)
            clip_max = offset - 1
            clip_min = -offset
            scale = (max_val - min_val) / (clip_max - clip_min)
            q_weight = torch.round((weight - min_val) / scale) + clip_min
            q_weight = torch.clip(q_weight, clip_min, clip_max)
            dq_weight = (q_weight - clip_min) * scale + min_val
            dq_weight = dq_weight.reshape(oc, ic).float()
            linear.weight.data = dq_weight
            return linear
        with torch.no_grad():
            for i in range(self.config.num_hidden_layers):
                for name, child in self.model.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].self_attn, name, quant_dequant(child))
                for name, child in self.model.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].mlp, name, quant_dequant(child))
            self.model.lm.lm = quant_dequant(self.model.lm.lm)

    def unload_param(self):
        self.unloaded_ops = {}
        self.experts = []
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            for i in range(len(self.model.blocks)):
                # different kv cache shape in different layers
                # if isinstance(self.config.num_attention_heads, list):
                self.model.blocks[i].self_attn.export_fused_attn = True
                is_moe = hasattr(self.model.blocks[i].mlp, 'is_moe') and self.model.blocks[i].mlp.is_moe
                if is_moe:
                    self.model.blocks[i].mlp.export_moe = True
                for name, child in self.model.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].self_attn, name, build_faker(child, f'/layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.model.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].mlp, name, build_faker(child, f'/layers.{i}/mlp/{name}/Linear'))
                    if name == 'shared_expert':
                        for name, child in self.model.blocks[i].mlp.shared_expert.named_children():
                            if isinstance(child, torch.nn.Linear):
                                setattr(self.model.blocks[i].mlp.shared_expert, name, build_faker(child, f'/layers.{i}/mlp/shared_expert/{name}/Linear'))
                    if is_moe and isinstance(child, torch.nn.ModuleList): # experts
                        self.experts.append(child)
                        for j in range(len(child)):
                            for name, cchild in child[j].named_children():
                                if isinstance(cchild, torch.nn.Linear):
                                    setattr(self.model.blocks[i].mlp.experts[j], name, build_faker(cchild, f'/expert/{i}_{j}/{name}'))
            self.model.lm.lm = build_faker(self.model.lm.lm, f'/lm/lm_head/Linear')

    @spinner_run(f'export model weight to ')
    def onnx_load_param(self, onnx_path):
        return OnnxRebuilder(onnx_path, self.unloaded_ops).rebuild()

    @spinner_run(f'slim the graph of ')
    def slim_onnx(self, onnx_model):
        import onnxslim
        model = onnxslim.slim(onnx_model)
        onnx.save(model, onnx_model)
        return onnx_model

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        # unload linear weight to save export memory
        self.unload_param()
        model = self.model
        seq_len = 3
        new_tokens = 0
        input_ids = torch.arange(seq_len, dtype=torch.long)
        attention_mask =  model.get_attention_mask(seq_len, new_tokens)
        position_ids = model.get_position_ids(seq_len, new_tokens, input_ids)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        # For export onnx, don't need image or audio's embedding
        input_ids = model.embedding(input_ids)
        logits_index = torch.tensor([-1], dtype=torch.int32)
        if hasattr(model, 'talker') and model.talker is not None:
            output_names = ['logits', 'hidden_states', 'talker_embeds']
        else:
            output_names = ['logits', 'hidden_states']

        # Qwen3-VL
        if self.model_type in ['qwen3_vl', 'qwen3_vl_moe']:
            # add deepstack_embeds input
            deepstack_embeds = torch.randn(3, 1, self.config.hidden_size)
            onnx_export(
                model, (input_ids, attention_mask, position_ids, logits_index, deepstack_embeds),
                onnx_model,
                input_names=[
                    'input_ids', 'attention_mask', 'position_ids', 'logits_index', 'deepstack_embeds'
                ],
                output_names=output_names,
                dynamic_axes=self.model_dynamic_axes)
            return onnx_model

        # export to onnx
        onnx_export(
            model, (input_ids, attention_mask, position_ids, logits_index),
            onnx_model,
            input_names=[
                'input_ids', 'attention_mask', 'position_ids', 'logits_index'
            ],
            output_names=output_names,
            dynamic_axes=self.model_dynamic_axes)
        return onnx_model

    def awq_quant(self):
        self.awq_quantizer = AwqQuantizer(self.model)
        self.awq_quantizer.quantize()

    def omni_quant(self):
        default_samples = 128
        total_lines = default_samples

        if self.args.calib_data:
            print(f"检测到 calib_data: {self.args.calib_data}，开始读取...")
            self.model.args.calib_data = self.args.calib_data

            if os.path.exists(self.args.calib_data):
                with open(self.args.calib_data, 'r', encoding='utf-8') as f:
                    # 统计总行数
                    total_lines = sum(1 for _ in f)
            else:
                print(f"错误：找不到文件 {self.args.calib_data}")

        calib_samples = min(total_lines, default_samples)

        print(f"OmniQuant 将使用 {calib_samples} 个样本进行优化 (Epochs={getattr(self.args, 'omni_epochs', 20)})...")

        self.omni_quantizer = OmniQuantizer(
            model=self.model,
            max_calib_samples=calib_samples,
            act_bit=self.args.act_bit,
            act_sym=self.args.act_sym,
            generate_for_npu=self.args.generate_for_npu,

            epochs=getattr(self.args, 'omni_epochs', 20),
            lr=getattr(self.args, 'omni_lr', 5e-3),
            wd=getattr(self.args, 'omni_wd', 1e-4)
        )
        self.omni_quantizer.quantize(self.args.generate_for_npu)

    def smooth_quant(self):
        total_lines = 128
        if self.args.calib_data:
            print(f"检测到 calib_data: {self.args.calib_data}，开始读取...")
            self.model.args.calib_data = self.args.calib_data

            if os.path.exists(self.args.calib_data):
                with open(self.args.calib_data, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
            else:
                print(f"错误：找不到文件 {self.args.calib_data}")

        calib_samples = min(total_lines, 128)
        self.smooth_quantizer = SmoothQuantizer(model = self.model, max_calib_samples = calib_samples, act_bit=self.args.act_bit, act_sym=self.args.act_sym, generate_for_npu=self.args.generate_for_npu)
        self.smooth_quantizer.quantize()

    def export_vision(self):
        if self.visual is None:
            return
        vision_onnx = self.visual.export(self.onnx_path)
        if self.mnn_converter:
            fuse_transformer = self.visual.transformer_fuse
            native_group_conv = self.visual.group_conv_native
            quant_bit_visual = self.visual.quant_bit
            quant_block_visual = self.visual.quant_block
            if self.args.transformer_fuse:
                fuse_transformer = True
            if self.args.group_conv_native:
                native_group_conv = True
            if self.args.visual_quant_bit is not None:
                quant_bit_visual = self.args.visual_quant_bit
            if self.args.visual_quant_block is not None:
                quant_block_visual = self.args.visual_quant_block
            self.mnn_converter.export(vision_onnx, quant_bit_visual,
                                      quant_block_visual,
                                      transformer_fuse=fuse_transformer,
                                      group_conv_native=native_group_conv,
                                      weight_sym=self.args.visual_sym)

    def export_audio(self):
        if self.audio is None:
            return
        audio_onnx = self.audio.export(self.onnx_path)
        if self.mnn_converter: self.mnn_converter.export(audio_onnx, self.audio.quant_bit)

    def export_talker(self):
        if self.talker is None:
            return
        talker_onnx = self.talker.export(self.onnx_path)
        predit_onnx, dit_onnx, bigvgan_onnx = self.talker.token2wav.export(self.onnx_path)
        if self.mnn_converter:
            self.mnn_converter.export(talker_onnx, self.talker.quant_bit)
            self.mnn_converter.export(predit_onnx, self.talker.token2wav.quant_bit)
            self.mnn_converter.export(dit_onnx, self.talker.token2wav.quant_bit)
            self.mnn_converter.export(bigvgan_onnx, self.talker.token2wav.quant_bit)

    def export_language(self):
        # export_embedding
        if self.mnn_converter and self.args.tie_word_embeddings:
            pass # mnn tie_word_embeddings need't export embedding
        else:
            self.export_embed()
        # export transformer
        onnx_model = self.export_onnx()

        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if self.mnn_converter:
            tie_embeddings_info = MNNConverter(self, self.unloaded_ops).export(onnx_model)
            if tie_embeddings_info is not None:
                self.llm_config['tie_embeddings'] = tie_embeddings_info
        else:
            self.onnx_load_param(onnx_model)

    def export(self, export_type):
        if not self.args.skip_weight:
            if self.args.omni:
                self.omni_quant()
            if self.args.awq:
                self.awq_quant()
            if self.args.smooth:
                self.smooth_quant()
        export_mnn = export_type == 'mnn'
        self.mnn_converter = MNNConverter(self) if export_mnn else None
        self.export_talker()
        self.export_vision()
        self.export_audio()
        self.export_eagle()
        self.export_language()
        self.export_mtp()
        self.export_tokenizer()
        self.export_config(export_mnn)
        if export_mnn:
            # delete onnx file
            try:
                for file in glob.glob(f'{self.onnx_path}/*'):
                    os.remove(file)
                os.rmdir(self.onnx_path)
            except Exception as e:
                print(f"remove onnx error: {e}")

    @spinner_run(f'export tokenizer to ')
    def export_tokenizer(self):
        return self.tokenizer.export(self.args.dst_path)

class EmbeddingExporter(LlmExporter):
    def __init__(self, args):
        super().__init__(args)

    def response(self, query):
        self.model.eval()
        prompt = self.build_prompt(query)
        input_ids = self.tokenizer(prompt)['input_ids']
        seq_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        res = self.model.forward(inputs_embeds, attention_mask, position_ids)
        print(res, res.shape)
        return res

    def build_prompt(self, content):
        if self.config.model_type == 'bert':
            return f'[CLS]{content}[SEP]'
        if self.config.model_type == 'new':
            return f'<s> {content}</s>'
        if self.config.model_type == 'qwen3':
            return f'{content}<|endoftext|>'

    @spinner_run(f'load pretrained model ', True)
    def load_model(self, model_path):
        self.model = EmbeddingModel.from_pretrained(model_path, args=self.args)
        self.config = self.model.config
        self.model_type = self.config.model_type
        self.tokenizer = LlmTokenizer(model_path, self.model_type)
        self.llm_config = {
            'model_type': self.config.model_type,
            'hidden_size' : self.config.hidden_size,
            'attention_mask': 'int',
            "jinja": {
                "chat_template": self.build_prompt("{{ messages | map(attribute='content') | join('') }}")
            },
            'is_visual': False
        }
        return model_path

    def export_reranker(self):
        seq_len = 4
        input_ids = torch.arange(12, dtype=torch.long)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        inputs_embeds = inputs_embeds.reshape(3, 4, self.config.hidden_size)
        attention_mask = torch.zeros(3, 1, 1, 4, dtype=torch.float)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        onnx_export(
            self.model, (inputs_embeds, attention_mask, position_ids),
            onnx_model,
            input_names=[
                'input_ids',
                'attention_mask',
                'position_ids'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes={
                "input_ids" : { 0: "batch", 1: "seq_len" },
                "position_ids" : { 1: "seq_len" },
                "attention_mask" : { 0: "batch", 3: "seq_len" }
            })
        return onnx_model

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        if self.model_type == 'qwen3':
            self.unload_param()
        else:
            self.unloaded_ops = None
        if self.model.is_reranker:
            return self.export_reranker()
        seq_len = 3
        input_ids = torch.arange(seq_len, dtype=torch.long)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        onnx_export(
            self.model, (inputs_embeds, attention_mask, position_ids),
            onnx_model,
            input_names=[
                'input_ids',
                'attention_mask',
                'position_ids'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes={
                "input_ids" : { 1: "seq_len" },
                "position_ids" : { 1: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" }
            })
        return onnx_model

    def export(self, export_type):
        export_mnn = 'mnn' in export_type
        self.export_tokenizer()
        self.export_embed()
        self.export_config(export_mnn)
        onnx_model = self.export_onnx()
        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if export_mnn:
            transformer_fuse = not self.model.is_reranker
            tie_embeddings_info = MNNConverter(self, self.unloaded_ops).export(onnx_model, transformer_fuse=transformer_fuse)
            if tie_embeddings_info is not None:
                self.llm_config['tie_embeddings'] = tie_embeddings_info
            # delete onnx file
            try:
                for file in glob.glob(f'{self.onnx_path}/*'):
                    os.remove(file)
                os.rmdir(self.onnx_path)
            except Exception as e:
                print(f"remove onnx error: {e}")

def build_args(parser):
    parser.add_argument('--path', type=str, required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, default=None,
                        help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer path, default is `None` mean using `--path` value.')
    parser.add_argument('--eagle_path', type=str, default=None, help='eagle model path, default is `None`')
    parser.add_argument('--lora_path', type=str, default=None, help='lora path, default is `None` mean not apply lora.')
    parser.add_argument('--gptq_path', type=str, default=None, help='gptq path, default is `None` mean not apply gptq.')
    parser.add_argument('--dst_path', type=str, default='./model', help='export onnx/mnn model to path, default is `./model`.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to print verbose.')
    parser.add_argument('--test', type=str, help='test model inference with query `TEST`.')
    parser.add_argument('--export', type=str, default=None, help='export model to an onnx/mnn model.')
    parser.add_argument('--onnx_slim', action='store_true', help='Whether or not to use onnx-slim.')
    parser.add_argument('--quant_bit', type=int, default=4, help='mnn quant bit, 4 or 8, default is 4.')
    parser.add_argument('--quant_block', type=int, default=64, help='mnn quant block, 0 mean channel-wise, default is 64.')
    parser.add_argument('--visual_quant_bit', type=int, default=None, help='mnn visual quant bit, 4 or 8, default is setting in utils/vision.py by different vit model.')
    parser.add_argument('--visual_quant_block', type=int, default=None, help='mnn quant block, default is setting in utils/vision.py by different vit model.')
    parser.add_argument('--lm_quant_bit', type=int, default=None, help='mnn lm_head quant bit, 4 or 8, default is `quant_bit`.')
    parser.add_argument('--lm_quant_block', type=int, default=None, help='mnn lm_head quant block, 0 mean channle-wise, default is `quant_block`.')
    parser.add_argument('--mnnconvert', type=str, default='../../../build/MNNConvert', help='local mnnconvert path, if invalid, using pymnn.')
    parser.add_argument('--ppl', action='store_true', help='Whether or not to get all logits of input tokens.')
    parser.add_argument('--awq', action='store_true', help='Whether or not to use awq quant.')
    parser.add_argument('--hqq', action='store_true', help='Whether or not to use hqq quant.')
    parser.add_argument('--omni', action='store_true', help='Whether or not to use omni quant.')
    parser.add_argument('--transformer_fuse', action='store_true', help='Whether or not to fuse vision transformer op.')
    parser.add_argument('--group_conv_native', action='store_true', help='Whether or not to keep native group_conv.')
    parser.add_argument('--smooth', action='store_true', help='Whether or not to use smooth quant.')
    parser.add_argument('--sym', action='store_true', help='Whether or not to using symmetric quant (without zeropoint), default is False.')
    parser.add_argument('--visual_sym', action='store_true', help='Whether or not to using symmetric quant (without zeropoint) for visual model, default is False.')
    parser.add_argument('--seperate_embed', action='store_true', help='For lm and embed shared model, whether or not to sepearte embed to avoid quant, default is False, if True, embed weight will be seperate to embedding bf16.bin.')
    parser.add_argument('--lora_split', action='store_true', help='Whether or not export lora split, default is False.')
    parser.add_argument('--calib_data', type=str, default=None, help='calibration data path, default is `None` mean not use calib data.')
    parser.add_argument('--act_bit', type=int, default=16, help='smooth quant act bit, 8 or 16, default is 16.')
    parser.add_argument('--embed_bit', type=int, default=16, choices=[16, 8, 4], help='embedding export bit precision, choices are 16 (bf16), 8 (int8), 4 (int4), default is 16.')
    parser.add_argument('--act_sym', action='store_true', help='smooth quant act us sym or not, default asym.')
    parser.add_argument('--quant_config', type=str, default=None, help='path to the JSON file for op-wise quantization configuration.')
    parser.add_argument('--generate_for_npu', action='store_true', help='Whether or not to generate model for NPU deployment, default is False.')
    parser.add_argument('--skip_weight', action='store_true', help='Whether or not to skip loading model weights, useful for testing export flow.')
    # omni quant
    parser.add_argument('--omni_epochs', type=int, default=20, help='OmniQuant 优化的轮数')
    parser.add_argument('--omni_lr', type=float, default=5e-3, help='OmniQuant 的学习率')
    parser.add_argument('--omni_wd', type=float, default=1e-4, help='OmniQuant 的权重衰减')

def export(path, **kwargs):
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args(['--path', path])
    for k, v in kwargs.items():
        setattr(args, k, v)
    if 'bge' in path:
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)
    # export
    llm_exporter.export(args.export)

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    build_args(parser)
    args = parser.parse_args()

    model_path = args.path

    embedding_models = ['bge', 'gte', 'Qwen3-Embedding']
    if any(model in model_path for model in embedding_models):
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)

    # some actions
    if args.test is not None:
        llm_exporter.response(args.test)

    if args.export is not None:
        llm_exporter.export(args.export)

if __name__ == '__main__':
    main()