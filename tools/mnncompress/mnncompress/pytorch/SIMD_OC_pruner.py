from __future__ import print_function
import torch
import yaml
import numpy as np

from ..common import MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
import uuid

class SIMDOCPruner(object):
    def __init__(self, model, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info=False, prune_finetune_iterations=0, max_prune_ratio=0.99):
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
        self._total_pruning_iterations = total_pruning_iterations
        self._prune_finetune_iterations = prune_finetune_iterations
        self._debug = debug_info
        self._config_file = config_file
        self._pname_accgrad = {}
        self._pname_gt_reshape_mean = {}
        self._pname_gt_remains_reshape_mean = {}
        self._pname_mask = {}
        self._all_mask_grad_flatten = []
        self._config_dict = {}
        self._generated_config_dict = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0
    
    def _init(self):
        for module_name, module in self._model.named_modules():
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
                for n, p in module.named_parameters():
                    if 'weight' == n:
                        pname = module_name + "." + n
                        if p.grad is None:
                            continue
                        self._pname_accgrad[pname] = torch.zeros_like(p.grad)
                        self._pname_mask[pname] = torch.ones_like(p.data)
    
    def current_prune_ratios(self):
        prune_ratios = {}
        total_weight_num = 0
        pruned_weight_num = 0
        for n, m in self._pname_mask.items():
            ratio = 1 - np.mean(m.cpu().numpy())
            prune_ratios[n] = ratio
            total_weight_num += m.numel()
            pruned_weight_num += int(m.numel() * ratio)
        
        prune_ratios['overall_prune_ratio'] = float(pruned_weight_num) / total_weight_num

        return prune_ratios

    def do_pruning(self, result_file = "SIMDOC_found_prune_ratios.yml"):
        if self._current_accumulate_step == 0:
            self._init()
        
        if self._current_accumulate_step < self._total_accumulate_steps:
            print("SIMDOCPruner: accumulating weight sensitivity...")
            for n, p in self._model.named_parameters():
                if n in self._pname_accgrad.keys():
                    self._pname_accgrad[n] += torch.abs(p.grad * p.data)

            self._current_accumulate_step += 1

        elif self._current_accumulate_step == self._total_accumulate_steps:
            # named_parameters = dict(self._model.named_parameters())
            # for n, g in self._pname_accgrad.items():
            #     self._pname_accgrad[n] = torch.abs(named_parameters[n]) * g

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
        algorithm.prune_params.type = compress_pb.PruneParams.SIMD_OC
        weight_tensor_names = algorithm.prune_params.simd_oc_pruner_params.weight_tensor_names
        layer_names = algorithm.prune_params.simd_oc_pruner_params.layer_names
        prune_ratios = algorithm.prune_params.simd_oc_pruner_params.prune_ratios
        oc_blocks = algorithm.prune_params.simd_oc_pruner_params.oc_blocks

        for n, m in self._pname_mask.items():
            # layer_names.append(op_name)
            weight_tensor_names.append(n)
            ratio = 1 - np.mean(m.cpu().numpy())
            self._total_weight_num += m.numel()
            self._remain_weight_num += (m.numel() * (1 - ratio) / 4.0)
            prune_ratios.append(ratio)
            oc_blocks.append(4)
            print(n, "pruning_ratio:", ratio)

        if not self._reported:
            detail = {"algorithm": "SIMD_OC", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"overall_sparsity": self._target_sparsity, "total_pruning_iterations": self._total_pruning_iterations, "prune_ratios": self._config_dict}}
            self._reported = mnn_logger.on_done("pytorch", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

    def _generate_prune_ratios(self, config_dict = {}):
        all_mask_grad_flatten = []
        for n, g in self._pname_accgrad.items():
            gn = g.detach().cpu().numpy()
            dims = len(gn.shape)
            trans_dims = [x for x in range(1, dims)] + [0]
            g_trans = gn.transpose(trans_dims)
            g_trans_shape = g_trans.shape
            oc_blocks = g_trans_shape[-1] // 4
            remains = g_trans_shape[-1] % 4
            block_total_rows = oc_blocks * 4

            if oc_blocks != 0:
                gt_reshape = g_trans[..., 0:block_total_rows].reshape((-1, 4))
                gt_reshape_mean = gt_reshape.mean(axis=1, keepdims=True)
                self._pname_gt_reshape_mean[n] = gt_reshape_mean
                all_mask_grad_flatten.append(gt_reshape_mean.flatten())

            if remains > 0:
                gt_remains_reshape = g_trans[..., block_total_rows:block_total_rows+remains].reshape((-1, remains))
                gt_remains_reshape_mean = gt_remains_reshape.mean(axis=1, keepdims=True)
                self._pname_gt_remains_reshape_mean[n] = gt_remains_reshape_mean
                all_mask_grad_flatten.append(gt_remains_reshape_mean.flatten())

        all_mask_grad_flatten = np.concatenate(all_mask_grad_flatten)
        all_mask_grad_flatten = np.sort(all_mask_grad_flatten)
        threshold_index = int(all_mask_grad_flatten.size * self._target_sparsity)
        threshold = all_mask_grad_flatten[threshold_index]
        self._all_mask_grad_flatten = all_mask_grad_flatten

        total_weight_num = 0
        total_pruned_weight_num = 0
        nm = dict(self._model.named_modules())
        for n, g in self._pname_accgrad.items():
            if n in self._pname_gt_reshape_mean.keys():
                gt_reshape_mean = self._pname_gt_reshape_mean[n]
                mask = gt_reshape_mean > threshold
                total_weight_num += g.numel()
                pruned_num = np.sum((mask == False)) * 4
            
            if n in self._pname_gt_remains_reshape_mean.keys():
                gt_remains_reshape_mean = self._pname_gt_remains_reshape_mean[n]
                mask = gt_remains_reshape_mean > threshold
                pruned_num += (np.sum((mask == False)) * (g.shape[0] % 4))

            sparsity = float(pruned_num) / g.numel()
            self._generated_config_dict[n] = sparsity
            if n in config_dict.keys():
                self._config_dict[n] = config_dict[n]
            else:
                self._config_dict[n] = sparsity
            
            if self._config_dict[n] > self._max_prune_ratio:
                self._config_dict[n] = self._max_prune_ratio

            module_name = n[:-7]
            if isinstance(nm[module_name], torch.nn.Conv2d) and nm[module_name].groups > 1:
                self._config_dict[n] = 0.0
            
            total_pruned_weight_num += int(g.numel() * self._config_dict[n])

        self._target_sparsity = float(total_pruned_weight_num) / total_weight_num
        print("overall prune ratio:", self._target_sparsity)
        print("pruning config:")
        for key, value in self._config_dict.items():
            print(key, ":", value)

    def _generate_prune_mask(self):
        if self._step > self._total_pruning_iterations:
            return
        
        threshold_index = int(self._all_mask_grad_flatten.size * self._target_sparsity * self._step / self._total_pruning_iterations)
        threshold = self._all_mask_grad_flatten[threshold_index]

        total_weight_num = 0
        total_pruned_weight_num = 0
        config_dict = {}

        for n, g in self._pname_accgrad.items():
            mask = None
            if n in self._pname_gt_reshape_mean.keys():
                gt_reshape_mean = self._pname_gt_reshape_mean[n]
                mask = gt_reshape_mean > threshold
                if np.abs(self._generated_config_dict[n] - self._config_dict[n]) > 0.01:
                    argsorted = np.argsort(gt_reshape_mean.flatten())
                    local_threshold_index = int(gt_reshape_mean.size * self._config_dict[n] * self._step / self._total_pruning_iterations)
                    mask = np.ones_like(gt_reshape_mean.flatten())
                    mask[argsorted[0:local_threshold_index]] = 0.
                    mask = mask.reshape(gt_reshape_mean.shape)
                mask = np.concatenate([mask, mask, mask, mask], axis=1)
                oc_blocks = g.shape[0] // 4
                block_total_rows = oc_blocks * 4
                mask = mask.reshape((-1, block_total_rows))

            if n in self._pname_gt_remains_reshape_mean.keys():
                gt_remains_reshape_mean = self._pname_gt_remains_reshape_mean[n]
                remain_mask = gt_remains_reshape_mean > threshold
                if np.abs(self._generated_config_dict[n] - self._config_dict[n]) > 0.01:
                    argsorted = np.argsort(gt_remains_reshape_mean.flatten())
                    local_threshold_index = int(gt_remains_reshape_mean.size * self._config_dict[n] * self._step / self._total_pruning_iterations)
                    remain_mask = np.ones_like(gt_remains_reshape_mean.flatten())
                    remain_mask[argsorted[0:local_threshold_index]] = 0.
                    remain_mask = remain_mask.reshape(gt_remains_reshape_mean.shape)
                remains = g.shape[0] % 4
                remain_mask = np.concatenate([remain_mask for x in range(0, remains)], axis=1)
                if mask is not None:
                    mask = np.concatenate([mask, remain_mask], axis=1)
                else:
                    mask = remain_mask

            g_shape = np.array([x for x in g.shape])
            reshape_shape = np.concatenate([g_shape[1:], g_shape[0:1]])
            mask = mask.reshape(reshape_shape)
            dims = len(g.shape)
            trans_dims = [dims-1] + [x for x in range(0, dims-1)]
            mask = mask.transpose(trans_dims)

            if self._debug:
                total_weight_num += mask.size
                total_pruned_weight_num += np.sum((mask == False))
                sparsity = 1.0 - np.mean(mask)
                config_dict[n] = sparsity.tolist()
            self._pname_mask[n] = torch.Tensor(mask).to(g.device)

        if self._debug:
            print("prune step:", self._step, "total prune steps:", self._total_pruning_iterations)
            print("overall prune ratio:", float(total_pruned_weight_num) / total_weight_num)
            print("pruning config:")
            for key, value in config_dict.items():
                print(key, ":", value)

    # def _generate_prune_mask(self):
    #     total_weight_num = 0
    #     total_pruned_weight_num = 0
    #     config_dict = {}

    #     for n, g in self._pname_accgrad.items():
    #         mask = None
    #         if n in self._pname_gt_reshape_mean.keys():
    #             gt_reshape_mean = self._pname_gt_reshape_mean[n]
    #             gt_reshape_mean_argsort = np.argsort(gt_reshape_mean.flatten())
    #             final_threshold_index = int(gt_reshape_mean.size * self._config_dict[n])
    #             threshold_index = int(gt_reshape_mean.size * (self._config_dict[n] * self._step / self._total_pruning_iterations))

    #             mask = np.ones(gt_reshape_mean_argsort.shape)
    #             mask[0:threshold_index] = 0.
    #             mask = mask.reshape(gt_reshape_mean.shape)
    #             mask = np.concatenate([mask, mask, mask, mask], axis=1)
    #             oc_blocks = g.shape[0] // 4
    #             block_total_rows = oc_blocks * 4
    #             mask = mask.reshape((-1, block_total_rows))

    #         if n in self._pname_gt_remains_reshape_mean.keys():
    #             gt_remains_reshape_mean = self._pname_gt_remains_reshape_mean[n]
    #             gt_remains_reshape_mean_argsort = np.argsort(gt_remains_reshape_mean.flatten())
    #             final_threshold_index = int(gt_remains_reshape_mean.size * self._config_dict[n])
    #             threshold_index = int(gt_remains_reshape_mean.size * (self._config_dict[n] * self._step / self._total_pruning_iterations))
    #             remain_mask = np.ones(gt_remains_reshape_mean_argsort.shape)
    #             remain_mask[0:threshold_index] = 0.
    #             remain_mask = remain_mask.reshape(gt_remains_reshape_mean.shape)
    #             remains = g.shape[0] % 4
    #             remain_mask = np.concatenate([remain_mask for x in range(0, remains)], axis=1)
    #             if mask is not None:
    #                 mask = np.concatenate([mask, remain_mask], axis=1)
    #             else:
    #                 mask = remain_mask

    #         g_shape = np.array([x for x in g.shape])
    #         reshape_shape = np.concatenate([g_shape[1:], g_shape[0:1]])
    #         mask = mask.reshape(reshape_shape)
    #         dims = len(g.shape)
    #         trans_dims = [dims-1] + [x for x in range(0, dims-1)]
    #         mask = mask.transpose(trans_dims)

    #         if self._debug:
    #             total_weight_num += mask.size
    #             total_pruned_weight_num += np.sum((1 - mask))
    #             sparsity = 1.0 - np.mean(mask)
    #             config_dict[n] = sparsity.tolist()
    #         self._pname_mask[n] = torch.Tensor(mask).to(g.device)

    #     if self._debug:
    #         print("prune step:", self._step, "total prune steps:", self._total_pruning_iterations)
    #         print("overall prune ratio:", float(total_pruned_weight_num) / total_weight_num)
    #         print("pruning config:")
    #         for key, value in config_dict.items():
    #             print(key, ":", value)
