from __future__ import print_function
import tensorflow as tf
from .SNIP_level_pruner import SNIPLevelPruner
from .SIMD_OC_pruner import SIMDOCPruner
from .TaylorFO_channel_pruner import TaylorFOChannelPruner
from .decomposition import low_rank_decompose
from .EMA_quantizer import EMAQuantizer
from .OAQ_quantizer import OAQQuantizer
from .LSQ_quantizer import LSQQuantizer


class SNIPLevelPrunerHook(tf.train.SessionRunHook):
    def __init__(self, sensitivity_npy_file, graph=None, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, prune_ratio_result_file="SNIPLevel_found_prune_ratios.yml"):
        self._pruner = SNIPLevelPruner(sensitivity_npy_file, graph, sparsity, total_pruning_iterations, config_file, debug_info, prune_finetune_iterations, max_prune_ratio, prune_ratio_result_file)
    
    def after_run(self, run_context, run_values):
        sess = run_context.session
        self._pruner.do_pruning(sess)


class SIMDOCPrunerHook(tf.train.SessionRunHook):
    def __init__(self, sensitivity_npy_file, graph=None, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, prune_ratio_result_file="SIMDOC_found_prune_ratios.yml"):
        self._pruner = SIMDOCPruner(sensitivity_npy_file, graph, sparsity, total_pruning_iterations, config_file, debug_info, prune_finetune_iterations, max_prune_ratio, prune_ratio_result_file)
    
    def after_run(self, run_context, run_values):
        sess = run_context.session
        self._pruner.do_pruning(sess)


class TaylorFOChannelPrunerHook(tf.train.SessionRunHook):
    def __init__(self, sensitivity_npy_file, graph=None, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, prune_ratio_result_file="TaylorFOChannel_found_prune_ratios.yml", align_channels=4):
        self._pruner = TaylorFOChannelPruner(sensitivity_npy_file, graph, sparsity, total_pruning_iterations, config_file, debug_info, prune_finetune_iterations, max_prune_ratio, prune_ratio_result_file, align_channels)
    
    def after_run(self, run_context, run_values):
        sess = run_context.session
        self._pruner.do_pruning(sess)

class LowRankDecompositionHook(tf.train.SessionHook):
    def __init__(self, graph, weight_npy_file, compress_params_file, skip_layers=[""], align_channels=8, tucker_minimal_ratio=0.25, reserved_singular_value_ratio=0.5, append=False):
        low_rank_decompose(graph, weight_npy_file, compress_params_file, skip_layers, align_channels, tucker_minimal_ratio, reserved_singular_value_ratio, append)

class EMAQuantizerHook(tf.train.SessionRunHook):
    def __init__(self, graph=None, is_training = None, skip_quant_layers = [], skip_quant_flag = [], bits = 8, debug_info = False, retain_sparsity=False):
        self.quantizer = EMAQuantizer(graph, is_training, skip_quant_layers, skip_quant_flag, bits, debug_info, retain_sparsity)
    
    def after_create_session(self, session, coord):
        self.quantizer.init(session)

    def after_run(self, run_context, run_values):
        sess = run_context.session
        self.quantizer.update(sess)

class OAQQuantizerHook(tf.train.SessionRunHook):
    def __init__(self, graph=None, is_training = None, skip_quant_layers = [], skip_quant_flag = [], bits = 8, min_clamp_value = 0, debug_info = False, retain_sparsity=False):
        self.quantizer = OAQQuantizer(graph, is_training, skip_quant_layers, skip_quant_flag, bits, min_clamp_value, debug_info, retain_sparsity)
    
    def after_create_session(self, session, coord):
        self.quantizer.init(session)

    def after_run(self, run_context, run_values):
        sess = run_context.session
        self.quantizer.update(sess)

class LSQQuantizerHook(tf.train.SessionRunHook):
    def __init__(self, graph=None, is_training = None, skip_quant_layers = [], skip_quant_flag = [], bits = 8, debug_info = False, retain_sparsity=False):
        self.quantizer = LSQQuantizer(graph, is_training, skip_quant_layers, skip_quant_flag, bits, debug_info, retain_sparsity)
    
    def after_create_session(self, session, coord):
        self.quantizer.init(session)

    def after_run(self, run_context, run_values):
        sess = run_context.session
        self.quantizer.update(sess)
