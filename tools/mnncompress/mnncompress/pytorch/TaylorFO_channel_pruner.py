from __future__ import print_function
import torch
try:
    import torch.fx as fx
except ImportError:
    print("need torch version >= 1.8.0, try 'pip install -U torch torchvision torchaudio' to upgrade torch")
import yaml
import numpy as np

from ..common import MNN_compression_pb2 as compress_pb
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
from .utils import find_conv_bn_module_pairs, not_safe_to_prune_weights
import uuid

class TaylorFOChannelPruner(object):
    def __init__(self, model, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, align_channels=4):
        self._model = model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model should be a torch.nn.Module instance")
        if sparsity < 0 or sparsity > 1:
            raise ValueError("sparsity should be a float number in [0, 1]")
        if total_pruning_iterations < 1:
            raise ValueError("total_pruning_iterations should be an integer >= 1")
        
        self._target_sparsity = sparsity
        self._max_prune_ratio = max_prune_ratio
        self._current_accumulate_step = 0
        self._total_accumulate_steps = 10
        self._step = 0
        self._finetune_step = 0
        self._config_file = config_file
        self._total_pruning_iterations = total_pruning_iterations
        self._prune_finetune_iterations = prune_finetune_iterations
        self._debug = debug_info
        self._pname_accgrad = {}
        self._pname_kernel_sens = {}
        self._pname_mask = {}
        self._pname_bias = {}
        self._pname_bn = {}
        self._config_dict = {}
        self._generated_config_dict = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0
        self._align_channels = align_channels
        self._not_safe_weights = []
        # self._not_safe_weights = not_safe_to_prune_weights(model)

    def _init(self):
        conv_bn_pairs = find_conv_bn_module_pairs(self._model)
        named_parameters = dict(self._model.named_parameters())
        for module_name, module in self._model.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                for n, p in module.named_parameters():
                    if 'weight' == n:
                        pname = module_name + "." + n
                        if p.grad is None:
                            continue
                        self._pname_accgrad[pname] = torch.zeros_like(p.grad)
                        self._pname_mask[pname] = torch.ones_like(torch.mean(p.data, [i for i in range(1, len(p.data.shape))], True))
                        
                        bias_name = module_name + "." + "bias"
                        if bias_name in named_parameters.keys():
                            self._pname_bias[pname] = named_parameters[bias_name]

                        if module_name in conv_bn_pairs.keys():
                            self._pname_bn[pname] = conv_bn_pairs[module_name]

    def current_prune_ratios(self):
        prune_ratios = {}
        total_weight_num = 0
        pruned_weight_num = 0
        for n, m in self._pname_mask.items():
            ratio = 1 - np.mean(m.cpu().numpy())
            prune_ratios[n] = ratio
            total_weight_num += self._pname_accgrad[n].numel()
            pruned_weight_num += int(self._pname_accgrad[n].numel() * ratio)
        
        prune_ratios['overall_prune_ratio'] = float(pruned_weight_num) / total_weight_num

        return prune_ratios

    def do_pruning(self, result_file = "TalayFO_found_prune_ratios.yml"):
        if self._current_accumulate_step == 0:
            self._init()
        
        if self._current_accumulate_step < self._total_accumulate_steps:
            print("TaylorFOChannelPruner: accumulating weight sensitivity...")
            for n, p in self._model.named_parameters():
                if n in self._pname_accgrad.keys():
                    self._pname_accgrad[n] += torch.abs(p.grad * p.data)

            self._current_accumulate_step += 1

        elif self._current_accumulate_step == self._total_accumulate_steps:
            config_dict = {}
            if self._config_file is not None:
                f = open(self._config_file, 'r')
                config_dict = yaml.safe_load(f)
            
            print("generating pruning ratios...")
            self._generate_prune_ratios(config_dict)
            
            f = open(result_file, "w")
            yaml.dump(self._config_dict, f)
            f.close()
            print("config_dict saved to file:", result_file)
            
            self._current_accumulate_step += 1
        
        else:
            if self._step < self._total_pruning_iterations:
                if self._prune_finetune_iterations == 0:
                    self._step += 1
                else:
                    if self._finetune_step == 0:
                        self._step += 1
                        self._finetune_step += 1
                    else:
                        self._finetune_step = (self._finetune_step + 1) % self._prune_finetune_iterations

            self._generate_prune_mask()
            for n, p in self._model.named_parameters():
                if n in self._pname_accgrad.keys():
                    p.data.mul_(self._pname_mask[n])
                
                if n in self._pname_bias.keys():
                    self._pname_bias[n].data.mul_(torch.flatten(self._pname_mask[n]))

                if n in self._pname_bn.keys():
                    bn_module_name = self._pname_bn[n]
                    bn = dict(self._model.named_modules())[bn_module_name]
                    if bn.weight is not None:
                        bn.weight.data.mul_(torch.flatten(self._pname_mask[n]))
                        bn.bias.data.mul_(torch.flatten(self._pname_mask[n]))
                    if bn.running_mean is not None:
                        bn.running_mean = bn.running_mean * torch.flatten(self._pname_mask[n])

            if self._step == self._total_pruning_iterations:
                self._step += 1

    def save_compress_params(self, filename, append=False):
        compress_proto = compress_pb.Pipeline()

        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

            pop_index = []
            for i in range(len(compress_proto.algo)):
                if compress_proto.algo[i].type == compress_pb.CompressionAlgo.CompressionType.PRUNE:
                    pop_index.append(i)
            for i in pop_index:
                compress_proto.algo.pop(i)

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid
        algorithm = compress_proto.algo.add()
        algorithm.type = compress_pb.CompressionAlgo.CompressionType.PRUNE
        algorithm.prune_params.type = compress_pb.PruneParams.FILTER
        weight_tensor_names = algorithm.prune_params.level_pruner_params.weight_tensor_names
        layer_names = algorithm.prune_params.level_pruner_params.layer_names
        prune_ratios = algorithm.prune_params.level_pruner_params.prune_ratios

        for n, m in self._pname_mask.items():
            # layer_names.append(op_name)
            weight_tensor_names.append(n)
            ratio = 1 - np.mean(m.cpu().numpy())
            self._total_weight_num += self._pname_accgrad[n].numel()
            # / 4.0, for we always expect using weight quant or full quant after prune
            self._remain_weight_num += (self._pname_accgrad[n].numel() * (1 - ratio) / 4.0)
            prune_ratios.append(ratio)
            print(n, "pruning_ratio:", ratio)

        if not self._reported:
            detail = {"algorithm": "TaylorFO_channel", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"overall_sparsity": self._target_sparsity, "total_pruning_iterations": self._total_pruning_iterations, "prune_ratios": self._config_dict}}
            self._reported = mnn_logger.on_done("pytorch", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

    def _generate_prune_ratios(self, config_dict = {}):
        for n, g in self._pname_accgrad.items():
            self._pname_kernel_sens[n] = torch.sum(g*g, [i for i in range(1, len(g.shape))], True)

        all_mask_grad_flatten = []
        for n, g in self._pname_kernel_sens.items():
            all_mask_grad_flatten.append(g.flatten())
        all_mask_grad_flatten = torch.cat(all_mask_grad_flatten)
        all_mask_grad_flatten = torch.sort(all_mask_grad_flatten).values
        threshold_index = int(all_mask_grad_flatten.numel() * self._target_sparsity)
        threshold = all_mask_grad_flatten[threshold_index]

        total_weight_num = 0
        total_pruned_weight_num = 0
        nm = dict(self._model.named_modules())
        for n, g in self._pname_kernel_sens.items():
            mask = g >= threshold
            total_weight_num += self._pname_accgrad[n].numel()
            sparsity = 1.0 - np.mean(mask.cpu().numpy())
            self._generated_config_dict[n] = sparsity.tolist()
            if n in config_dict.keys():
                self._config_dict[n] = config_dict[n]
            else:
                self._config_dict[n] = sparsity.tolist()
            
            if self._config_dict[n] > self._max_prune_ratio:
                self._config_dict[n] = self._max_prune_ratio

            if n in self._not_safe_weights:
                self._config_dict[n] = 0.0

            module_name = n[:-7]
            if isinstance(nm[module_name], torch.nn.Conv2d) and nm[module_name].groups > 1:
                self._config_dict[n] = 0.0

            remain_channels = int(mask.numel() * (1.0 - self._config_dict[n]))
            if remain_channels != mask.numel():
                remain_channels = remain_channels - remain_channels % self._align_channels
                if remain_channels <= 0:
                    remain_channels = min(self._align_channels, mask.numel())
            
            self._config_dict[n] = (mask.numel() - remain_channels + 0.0) / mask.numel()
            
            total_pruned_weight_num += int(self._pname_accgrad[n].numel() * self._config_dict[n])

        self._target_sparsity = float(total_pruned_weight_num) / total_weight_num
        print("overall prune ratio:", self._target_sparsity)
        print("pruning config:")
        for key, value in self._config_dict.items():
            print(key, ":", value)

    def _generate_prune_mask(self):
        if self._step > self._total_pruning_iterations:
            return
        
        total_weight_num = 0
        total_pruned_weight_num = 0
        config_dict = {}

        for n, g in self._pname_kernel_sens.items():
            g_flat = g.flatten()
            g_flat_argsort = torch.argsort(g_flat)
            final_threshold_index = int(g_flat.numel() * self._config_dict[n])
            threshold_index = int(g_flat.numel() * (self._config_dict[n] - self._config_dict[n] * self._step / self._total_pruning_iterations))
            mask = torch.ones(g_flat.shape)
            mask[g_flat_argsort[threshold_index:final_threshold_index]] = 0.
            mask = mask.reshape(g.shape)
            if self._debug:
                total_weight_num += mask.numel()
                total_pruned_weight_num += torch.sum((1 - mask).int()).cpu().numpy()
                sparsity = 1.0 - np.mean(mask.cpu().numpy())
                config_dict[n] = sparsity.tolist()
            self._pname_mask[n] = mask.to(g.device)

        if self._debug:
            print("prune step:", self._step, "total prune steps:", self._total_pruning_iterations)
            print("overall prune ratio:", float(total_pruned_weight_num) / total_weight_num)
            print("pruning config:")
            for key, value in config_dict.items():
                print(key, ":", value)
