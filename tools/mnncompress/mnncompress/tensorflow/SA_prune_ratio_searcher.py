from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import yaml
from mnncompress.tensorflow.helpers import get_op_name_of_type, is_weight_tensor, get_variable_by_tensor

_supported_prune_algrithms = ['level']
_Prune_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']


class SAPruneRatioSearcher(object):
    def __init__(self, sess, performance_evaluate_func, \
                sparsity = 0.5, cool_down_rate=0.95, perturbation_magnitude=0.5, start_temperature=100., stop_temperature=10., \
                measurement_mode = 'maximize', prune_algorithm = 'level', debug_info = False):
        print("\n\nsparsity, start_temperature, stop_temperature, cool_down_rate, perturbation_magnitude:", sparsity, start_temperature, stop_temperature, cool_down_rate, perturbation_magnitude)
        self._sess = sess
        self._graph = self._sess.graph
        
        if sparsity >= 1.0:
            raise ValueError("sparsity must be a number in (0, 1)")
        self._target_sparsity = sparsity
        
        if stop_temperature < 0.1 or stop_temperature > start_temperature:
            raise ValueError("temperature settings should be: start_temperature > stop_temperature >= 0.1")
        self._start_temperature = float(start_temperature)
        self._current_temperature = tf.get_variable(name='current_temperature', initializer=self._start_temperature, trainable=False)
        self._sess.run(self._current_temperature.initializer)
        self._stop_temperature = float(stop_temperature)
        
        if cool_down_rate <= 0 or cool_down_rate >= 1:
            raise ValueError("cool_down_rate shoulde be between 0 and 1")
        self._cool_down_rate = cool_down_rate
        self._current_performance = -np.inf
        self._best_performance = -np.inf
        self._best_performance_iter = 1
        self._best_config_dict = {}

        if perturbation_magnitude <= 0 or perturbation_magnitude >= 1:
            raise ValueError("perturbation_magnitude should be between 0 and 1")
        self._perturbation_magnitude = perturbation_magnitude

        if prune_algorithm not in _supported_prune_algrithms:
            raise ValueError()
        self._prune_algorithm = prune_algorithm

        self._debug = debug_info
        if debug_info:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        
        self._performance_evaluate_func = performance_evaluate_func
        
        if measurement_mode not in ['maximize', 'minimize']:
            raise ValueError("measurement_mode must be either 'maximize' or 'minimize', which indicates the optimization direction of 'test_performance_node'")
        self._measurement_mode = measurement_mode

        self._prune_op_names = get_op_name_of_type(self._graph, _Prune_Support_Ops)
        self._all_weight_variable_op_pairs = self._get_all_weight_variable_op_pairs(self._prune_op_names)
        self._all_weight_variable_sorted = self._get_all_weight_variables_sorted()

        self._sorted_weight_variable_sizes = []
        for v in self._all_weight_variable_sorted:
            self._sorted_weight_variable_sizes.append(self._sess.run(v).size)
        self._sorted_weight_variable_sizes = np.array(self._sorted_weight_variable_sizes).astype(np.float32)
        
        self._sparsities = self._init_sparsity()
        self._perturbed_sparsities = self._generate_perturbed_sparsities()
        
        if self._debug:
            print_op = tf.print("perturbed_sparsities:", self._perturbed_sparsities)
            with tf.control_dependencies([print_op]):
                self._perturbed_sparsities = tf.identity(self._perturbed_sparsities)

        self._current_sparsities = self._get_current_sparsities()
        print("initial sparsities:", sess.run(self._current_sparsities))

        if self._debug:
            print_op = tf.print("current sparsities:", self._current_sparsities)
            with tf.control_dependencies([print_op]):
                self._current_sparsities = tf.identity(self._current_sparsities)
        
        self._update_sparsities = tf.assign(self._sparsities, self._current_sparsities)
        self._restore_backup_weights = self._get_restore_backup_weights_ops()
        self._apply_masks = self._generate_masks_according_to_sparsities(self._prune_algorithm)
        self._update_temperature = tf.assign(self._current_temperature, self._current_temperature * self._cool_down_rate)
        self._restart_temperature = tf.assign(self._current_temperature, self._start_temperature)
        self._prune_iteration = 1

    def search(self, result_file = "SA_found_prune_ratios.yml"):
        for t in range(2):
            self._sess.run(self._restart_temperature)
            self._search()

        print("\n\n\n----------search complete--------------")
        best_performance = self._best_performance
        if self._measurement_mode == 'minimize':
            best_performance = -best_performance
        print("best performance:", best_performance, ", found at iteration:", self._best_performance_iter)
        print("pruner config dict found:", self._best_config_dict)

        f = open(result_file, "w")
        yaml.dump(self._best_config_dict, f)
        f.close()
        print("config_dict saved to file:", result_file)
        
        return self._best_config_dict
    
    def _search(self):

        while self._sess.run(self._current_temperature) > self._stop_temperature:
            print('\n\nsearch iteration:', self._prune_iteration)
            print('current temperature, stop temperature:', self._sess.run(self._current_temperature), self._stop_temperature)

            while True:
                self._sess.run(self._apply_masks)
                evaluation_result = self._performance_evaluate_func()

                config_dict = self._generate_config_dict()
                print("current config dict:", config_dict)

                if self._measurement_mode == "minimize":
                    evaluation_result *= -1

                # if better evaluation result, then accept the perturbation
                if evaluation_result > self._current_performance:
                    print("better performance")
                    self._current_performance = evaluation_result
                    self._sess.run(self._update_sparsities)
                    self._sess.run(self._restore_backup_weights)

                    # save best performance and best params
                    if evaluation_result > self._best_performance:
                        print("update best model")
                        self._best_performance = evaluation_result
                        self._best_performance_iter = self._prune_iteration
                        self._best_config_dict = config_dict
                    break
                # if not, accept with probability e^(-deltaE/current_temperature)
                else:
                    delta_E = np.abs(evaluation_result - self._current_performance)
                    probability = math.exp(-1 * delta_E / self._sess.run(self._current_temperature))
                    random_num = np.random.uniform(0, 1)
                    if random_num < probability:
                        print("accept with probability", random_num, probability)
                        self._current_performance = evaluation_result
                        self._sess.run(self._update_sparsities)
                        self._sess.run(self._restore_backup_weights)
                        break
                
                self._sess.run(self._restore_backup_weights)

            print("current best config_dict: ", self._best_config_dict)

            # cool down
            self._sess.run(self._update_temperature)
            self._prune_iteration += 1

    def _get_current_sparsities(self):
        current_sparsities = []
        for i in range(len(self._all_weight_variable_sorted)):
            weight = self._all_weight_variable_sorted[i]
            weight_abs = tf.abs(weight)
            sparsity = tf.reduce_mean(tf.cast(weight_abs < 1e-8, tf.float32)) * tf.constant([1.])
            current_sparsities.append(sparsity)
        current_sparsities = tf.concat(current_sparsities, axis=0)

        return current_sparsities

    def _generate_config_dict(self):
        config_dict = {}
        current_sparsities = self._sess.run(self._current_sparsities).tolist()
        for i in range(len(self._all_weight_variable_sorted)):
            weight = self._all_weight_variable_sorted[i]
            op = self._all_weight_variable_op_pairs[weight]
            sparsity = current_sparsities[i]
            config_dict[op.name] = sparsity
        
        return config_dict

    def _generate_level_pruner_masks(self):
        apply_masks = []
        for i in range(len(self._all_weight_variable_sorted)):
            v = self._all_weight_variable_sorted[i]
            size = self._sorted_weight_variable_sizes[i]
            sparsity = self._perturbed_sparsities[i]
            
            sparse_num = size * sparsity
            sparse_num = tf.cast(sparse_num, tf.int64)
            weight_abs = tf.abs(v)
            weight_abs_sorted = tf.contrib.framework.sort(tf.reshape(weight_abs, [-1]))
            threshold = weight_abs_sorted[sparse_num]
            mask = tf.cast(weight_abs >= threshold, tf.float32)
            if self._debug:
                print_op = tf.print("mask sparsity:", 1-tf.reduce_mean(mask), sparsity)
                with tf.control_dependencies([print_op]):
                    mask = tf.identity(mask)
            apply_mask = tf.assign(v, v * mask)
            apply_masks.append(apply_mask)
        
        return apply_masks

    def _generate_masks_according_to_sparsities(self, prune_algorithm):
        if prune_algorithm == "level":
            apply_masks = self._generate_level_pruner_masks()
            return apply_masks
        else:
            raise NotImplementedError("prune algorithm: " + prune_algorithm + " not supported now")

    def _get_restore_backup_weights_ops(self):
        restore_weights_ops = []
        for v in self._all_weight_variable_sorted:
            restore_weights_ops.append(tf.assign(v, self._sess.run(v)))
        return restore_weights_ops

    def _generate_perturbed_sparsities(self):
        magnitude = self._current_temperature / self._start_temperature * self._perturbation_magnitude
        perturbation = tf.random.uniform([len(self._all_weight_variable_sorted)], -magnitude, magnitude)
        if self._debug:
            print_op = tf.print("perturbation, sparsities, sparsities + perturbation:", perturbation, self._sparsities, self._sparsities + perturbation)
            with tf.control_dependencies([print_op]):
                perturbation = tf.identity(perturbation)
        sparsities = tf.clip_by_value(self._sparsities + perturbation, 0, tf.reduce_max(self._sparsities + perturbation))
        sparsities = tf.contrib.framework.sort(sparsities)

        sparse_num = tf.reduce_sum(sparsities * self._sorted_weight_variable_sizes)
        overall_sparsity = sparse_num / tf.reduce_sum(self._sorted_weight_variable_sizes)
        scale_factor = self._target_sparsity / overall_sparsity
        perturbed_sparsities = scale_factor * sparsities

        if self._debug:
            print_op = tf.print("after sparsity rescale:", perturbed_sparsities)
            with tf.control_dependencies([print_op]):
                perturbed_sparsities = tf.identity(perturbed_sparsities)

        perturbed_sparsities_mask = tf.cast(perturbed_sparsities < 1.0, tf.float32)
        perturbed_sparsities = perturbed_sparsities * perturbed_sparsities_mask + (1.0 - perturbed_sparsities_mask) * self._target_sparsity

        if self._debug:
            print_op = tf.print("before, after perturbation:", self._sparsities, perturbed_sparsities)
            with tf.control_dependencies([print_op]):
                return tf.identity(perturbed_sparsities)
        else:
            return perturbed_sparsities

    def _init_sparsity(self):
        while True:
            sparsities = np.sort(np.random.uniform(0, 1, len(self._all_weight_variable_sorted)))
            sparse_num = np.sum(sparsities * self._sorted_weight_variable_sizes)
            overall_sparsity = sparse_num / np.sum(self._sorted_weight_variable_sizes)
            scale_factor = self._target_sparsity / overall_sparsity
            sparsities = (scale_factor * sparsities).astype(np.float32)
            if sparsities[0] >= 0 and sparsities[-1] < 1:
                sparsities = tf.get_variable(name="mnn_sa_sparsities", initializer=tf.constant(sparsities), trainable=False)
                self._sess.run(sparsities.initializer)
                return sparsities

    def _get_all_weight_variables_sorted(self):
        all_weight_variables = [v for v in self._all_weight_variable_op_pairs.keys()]
        sorted_weight_variables = sorted(all_weight_variables, key=lambda v: self._sess.run(v).size)
        return sorted_weight_variables

    def _get_all_weight_variable_op_pairs(self, op_names):
        prune_ops = [self._graph.get_operation_by_name(op_name) for op_name in op_names]
        all_weight_variables = {}
        for op in prune_ops:
            if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                weight_variable = get_variable_by_tensor(op.inputs[1])
                if not self._sess.run(tf.is_variable_initialized(weight_variable)):
                    raise ValueError("weight variable not initialized, please do prune ratio search from a checkpoint")
                all_weight_variables[weight_variable] = op
            if op.type == "MatMul":
                if "gradients" in op.name and "_grad" in op.name:
                    continue
                
                for t in op.inputs:
                    if is_weight_tensor(t):
                        weight_variable = get_variable_by_tensor(t)
                        if not self._sess.run(tf.is_variable_initialized(weight_variable)):
                            raise ValueError("weight variable not initialized, please do prune ratio search from a checkpoint")
                        all_weight_variables[weight_variable] = op
        return all_weight_variables
