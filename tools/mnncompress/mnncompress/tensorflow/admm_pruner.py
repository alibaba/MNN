from __future__ import print_function
import yaml
import tensorflow as tf
import numpy as np

from .helpers import kronecker_tf, is_weight_tensor
from ..common import MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
import uuid

class ADMMPruner:
    def __init__(self, session, config_file, admm_epoch=15, lr=0.01, lr_decay=0.1, sparsity_type='random', rho=0.001):
        """
            default constructor
        :param session: contains the graph, model definition
        :param config_file: path to the *.yaml file, which contains detailed prune ratios for the model weights
        :param rho: parameter of ADMM
        """
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}
        self.sess = session
        self.weights = tf.trainable_variables()
        self.masks = {}
        self.prune_ops = []
        self.get_config(config_file)
        self.admm_epoch = admm_epoch
        self.lr = lr
        self.lr_decay = lr_decay
        self.sparsity_type = sparsity_type
        self.prune_method = self.get_prune_method(sparsity_type)
    
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
        method_dict = {
            'random': compress_pb.PruneParams.RANDOM,
            'SIMD_row': compress_pb.PruneParams.SIMD_ROW,
            'column': compress_pb.PruneParams.COLUMN,
            'filter': compress_pb.PruneParams.FILTER,
        }
        algorithm.prune_params.type = method_dict[self.sparsity_type]
        weight_tensor_names = algorithm.prune_params.level_pruner_params.weight_tensor_names
        prune_ratios = algorithm.prune_params.level_pruner_params.prune_ratios

        for tensor_name in self.masks.keys():
            weight_tensor_names.append(tensor_name)
            ratio = 1.0 - np.mean(self.masks[tensor_name]).tolist()
            prune_ratios.append(ratio)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

    def get_prune_method(self, sparsity_type='random'):
        if sparsity_type == 'random':
            # self.prune_method = self.random_prune
            prune_method = self.random_prune_npy
        elif sparsity_type == 'SIMD_row':
            # self.prune_method = self.SIMD_row_prune
            prune_method = self.SIMD_row_prune_npy
        elif sparsity_type == 'column':
            # self.prune_method = self.column_prune
            prune_method = self.column_prune_npy
        elif sparsity_type == 'filter':
            # self.prune_method = self.filter_prune
            prune_method = self.filter_prune_npy
        else:
            raise NotImplementedError
        return prune_method

    def get_config(self, config):
        """
            called by ADMM constructor. config should be a .yaml file
        :param config: configuration file that has settings for prune ratios, rhos

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                prs = raw_dict['prune_ratios']
                self.prune_ratios = {}
                for k, v in prs.items():
                    op = self.sess.graph.get_operation_by_name(k)
                    if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                        weight_tensor = op.inputs[1]
                        self.prune_ratios[weight_tensor.name] = v
                    elif op.type == 'MatMul':
                        input1 = op.inputs[0]
                        input2 = op.inputs[1]
                        if is_weight_tensor(input1):
                            self.prune_ratios[input1.name] = v
                        elif is_weight_tensor(input2):
                            self.prune_ratios[input2.name] = v
                        else:
                            print("WARNING:", k, "of type:", op.type, "has no weight, skip")
                    else:
                        raise ValueError(k + " of type: " + op.type + " is not supported")

                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho

                variables = self.weights
                np_values = self.sess.run(variables)
                for var, value in zip(variables, np_values):
                    if var.name not in self.prune_ratios:
                        continue
                    self.ADMM_U[var.name] = tf.Variable(np.zeros(shape=var.shape), dtype=float, trainable=False,
                                                        name='admm_U')
                    self.ADMM_Z[var.name] = tf.Variable(value, trainable=False, name='admm_Z')
                self.sess.run(tf.initialize_variables(self.ADMM_U.values(), name='init_U'))
                self.sess.run(tf.initialize_variables(self.ADMM_Z.values(), name='init_Z'))
                print("Prune ratios from .yaml")
                for k, v in self.prune_ratios.items():
                    print("      {} : {}".format(k, v))

            except yaml.YAMLError as exc:
                print(exc)

    def Z_initialize(self):
        """
            initialize the Z variables for each weight
        """
        variables = self.weights
        np_values = self.sess.run(variables)
        update_ops = []
        # for var in variables:
        for var, value in zip(variables, np_values):
            if var.name in self.prune_ratios:
                # Z(k+1) = W(k+1)+U(k)  U(k) is zeros here
                # _, updated_Z = self.prune_method(var, self.prune_ratios[var.name], group=4, norm='l2')
                _, updated_Z = self.prune_method(value, self.prune_ratios[var.name], group=4, norm='l2')
                update_ops.append(tf.assign(self.ADMM_Z[var.name], updated_Z))
        self.sess.run(update_ops)

    def Z_U_update(self, epoch, batch_idx, verbose=False, train_writer=None):
        """
            update Z and U, according to ADMM optimization with the SIMD pruning method
        :param device:
        :param train_loader:
        :param optimizer:
        :param epoch: number of epochs in the main loop
        :param data:
        :param batch_idx: index of the batch size
        :param writer: to print some information
        :return:
        """

        if epoch > 0 and epoch % self.admm_epoch == 0 and batch_idx == 0:
            print('--- ADMM iteration begin ---')
            variables = self.weights
            np_values = self.sess.run(variables)
            update_ops = []
            for var, value in zip(variables, np_values):
                if var.name not in self.prune_ratios:
                    continue
                Z_value, U_value = self.sess.run([self.ADMM_Z[var.name], self.ADMM_U[var.name]])

                Z_prev = Z_value
                # Z(k+1) = W(k+1) + U[k]
                Z_value = value + U_value

                # returns the mask and pruned weight
                # in mask, orginal values above threshold are 1, below are 0
                # pruned weight has percentile% channels set to 0
                mask_Z, updated_Z = self.prune_method(Z_value, self.prune_ratios[var.name], group=4, norm='l2')
                update_ops.append(tf.assign(self.ADMM_Z[var.name], updated_Z))

                if verbose:
                    if train_writer:
                        train_writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(var.name),
                                                     np.sqrt(np.sum((value - updated_Z) ** 2)), epoch)
                        train_writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(var.name),
                                                     np.sqrt(np.sum((updated_Z - Z_prev) ** 2)), epoch)

                # U(k+1) = W(k+1) - Z(k+1) +U(k)
                U_value = value - updated_Z + U_value
                update_ops.append(tf.assign(self.ADMM_U[var.name], U_value))

            self.sess.run(update_ops)
            print('--- ADMM iteartion end ---')

    def compute_loss(self):
        """
        :param ce_loss: the cross-entropy loss

        :return:
            ce_loss: original cross-entropy loss
            admm_loss: a dict to save loss for each layer
            mixed_loss: overall cross-entropy loss + admm loss
            total_U_norm: sum of norm(U, 2)
        """
        variables = self.weights
        admm_loss = tf.norm(0.)
        admm_U_norm = tf.norm(0.)
        for var in variables:
            if var.name not in self.prune_ratios:
                continue
            admm_loss += 0.5 * self.rhos[var.name] * tf.norm(var - self.ADMM_Z[var.name] + self.ADMM_U[var.name]) ** 2
            admm_U_norm += tf.norm(self.ADMM_U[var.name])

        return admm_loss, admm_U_norm

    def adjust_learning_rate(self, epoch):
        """
            re-adjust the learning rate in the ADMM optimization.
            Learning rate change is periodic. When epoch is dividable by admm_epoch, the learning rate is reset
            to the original value, and decay every 3 epoch
        :param epoch: number of epochs in the main loop
        :return:
        """
        lr = None
        if epoch % self.admm_epoch == 0:
            lr = self.lr
        else:
            offset = epoch % self.admm_epoch
            admm_step = self.admm_epoch / 3  # roughly every 1/3 admm_epoch
            lr = self.lr * (self.lr_decay ** (offset // admm_step))

        return lr

    def multi_rho_scheduler(self, name):
        """
        It works better to make rho monotonically increasing
        we increase it by 1.9x every admm epoch
        After 10 admm updates, the rho will be 0.91
        """

        self.rhos[name] *= 2

    def get_prune_ops(self, weights=None, masks=None):
        """
            hard prune on each weight of the model
        :return:
            prune ops
        """
        print('--- hard prune all weights ---')
        if not weights:
            variables = self.weights
        else:
            variables = weights

        if not masks:
            masks = self.masks

        self.prune_ops = []
        for var in variables:
            if var.name in self.prune_ratios:
                # print("name: {},\tshape: {}".format(var.name, var.shape))
                # prune expected weights by using masks
                self.prune_ops.append(tf.assign(var, tf.multiply(var, masks[var.name])))

        return self.prune_ops

    def hard_prune_weight(self, weights=None):
        """
            hard prune on each weight of the model
        :return:
            masks={name:value}, a dict of masks after pruning
        """
        print('--- hard prune all weights ---')
        if not weights:
            variables = self.weights
        else:
            variables = weights
        np_values = self.sess.run(variables)
        prune_ops = []
        for var, value in zip(variables, np_values):
            if var.name in self.prune_ratios:
                # print("name: {},\tshape: {}".format(var.name, var.shape))
                # SIMD prune on each weight
                mask, pruned_weight = self.prune_method(value, self.prune_ratios[var.name], group=4, norm='l2')
                # assign pruned weight back to the model
                prune_ops.append(tf.assign(var, pruned_weight))
                self.masks[var.name] = mask
        self.sess.run(prune_ops)
        return self.masks

    def SIMD_row_prune(self, weight, prune_ratio, group=4, norm='l1'):
        """
            the SIMD pruning method in the row direction
        :param weight: tensor of the weight from the model: (kernel, kernel, in-channel, out-channel)
        :param prune_ratio: float value, 0.00 - 1.00
        :param group: the number of elements in one group
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        if len(weight.shape) == 4:
            weight = tf.transpose(weight, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = tf.transpose(weight, (1, 0))

        shape = weight.shape
        print("SIMD row pruning {}: {}, length: {}".format(norm, weight.shape, group))
        weight2d = tf.reshape(weight, shape=(shape[0], -1))
        shape2d = weight2d.shape
        # deal with the last group if column is not divisable
        num_pad = (group - (weight2d.shape[1] % group)) % group
        rescale_factor = float(group) / int(group - num_pad)
        # padding zero to the last column
        weight2d_pad = tf.pad(weight2d, ((0, 0), (0, num_pad)), 'constant', constant_values=0)
        shape2d_pad = weight2d_pad.shape
        if norm == 'l1':
            weight_norm = tf.abs(weight2d_pad)
        elif norm == 'l2':
            weight_norm = tf.square(weight2d_pad)

        group_vals_matrix = None
        for i in range(shape2d_pad[1] // group):
            # sum weights of each group in row direction
            vector = tf.reduce_sum(weight_norm[:, i * group: (i + 1) * group], axis=1)  # l2 norm
            if i == (shape2d_pad[1] // group) - 1:
                # rescale the value of last group
                vector *= rescale_factor
            if i == 0:
                group_vals_matrix = tf.reshape(vector, (-1, 1))
            else:
                group_vals_matrix = tf.concat([group_vals_matrix, tf.reshape(vector, (-1, 1))], 1)
        group_vals_matrix = tf.sqrt(group_vals_matrix)

        # find the threshold
        group_vals_vec = tf.reshape(group_vals_matrix, (-1, 1))
        threshold = tf.gather(tf.contrib.framework.sort(group_vals_vec, axis=0), int(int(group_vals_vec.shape[0]) * prune_ratio))

        # set those > threshold as 1, those < threshold as 0
        zeros = tf.zeros(shape=group_vals_matrix.shape, dtype='float32')
        ones = tf.ones(shape=group_vals_matrix.shape, dtype='float32')
        group_vals_matrix = tf.where(tf.less_equal(group_vals_matrix, threshold), zeros, ones)
        # restore orignal shape by repeating 1/0 group_len times
        kron_matrix = tf.ones(shape=[1, group], dtype='float32')
        mask_padded = kronecker_tf(group_vals_matrix, kron_matrix)
        # delete paddings to restore original shape
        mask_final = mask_padded[:, : shape2d[1]]
        weight_masked = tf.multiply(weight2d, mask_final)
        weight_masked = tf.reshape(weight_masked, shape)
        mask_final = tf.reshape(mask_final, shape)

        if len(mask_final.shape) == 4:
            mask_final = tf.transpose(mask_final, (3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = tf.transpose(mask_final, (1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = tf.transpose(weight_masked, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight_masked = tf.transpose(weight_masked, (1, 0))

        return mask_final, weight_masked

    def SIMD_row_prune_npy(self, weight, prune_ratio, group=4, norm='l1'):
        """
            the SIMD pruning method in the row direction
        :param weight: numpy.array of the weight from the model
        :param prune_ratio: float value, 0.00 - 1.00
        :param group: the number of elements in one group
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        percent = prune_ratio * 100

        if len(weight.shape) == 4:
            weight = weight.transpose((3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = weight.transpose((1, 0))

        shape = weight.shape
        print("SIMD row pruning {}: {}, length: {}".format(norm, weight.shape, group))
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        # deal with the last group if column is not divisable
        num_pad = (group - (weight2d.shape[1] % group)) % group
        rescale_factor = float(group) / int(group - num_pad)
        # padding zero to the last column
        weight2d_pad = np.pad(weight2d, ((0, 0), (0, num_pad)), 'constant', constant_values=0)
        shape2d_pad = weight2d_pad.shape
        if norm == 'l1':
            weight_norm = np.abs(weight2d_pad)
        elif norm == 'l2':
            weight_norm = np.square(weight2d_pad)

        group_vals_matrix = np.zeros(shape=(shape2d_pad[0], shape2d_pad[1] // group), dtype=np.float32)
        for i in range(shape2d_pad[1] // group):
            # sum weights of each group in row direction
            vector = np.sum(weight_norm[:, i * group: (i + 1) * group], axis=1)  # l2 norm
            if i == (shape2d_pad[1] // group) - 1:
                # rescale the value of last group
                vector *= rescale_factor
            group_vals_matrix[:, i] = np.squeeze(vector)
        group_vals_matrix = np.sqrt(group_vals_matrix)

        # find the threshold
        threshold = np.percentile(group_vals_matrix, percent)
        # set those > threshold as 1, those < threshold as 0
        group_vals_matrix = np.where(group_vals_matrix <= threshold, 0, 1)
        # restore orignal shape by repeating 1/0 group_len times
        kron_matrix = np.ones(group)
        mask_padded = np.kron(group_vals_matrix, kron_matrix)
        # delete paddings to restore original shape
        mask_final = mask_padded[:, : shape2d[1]]
        weight_masked = weight2d * mask_final
        weight_masked = weight_masked.reshape(shape)
        mask_final = mask_final.reshape(shape)

        if len(mask_final.shape) == 4:
            mask_final = mask_final.transpose((3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = mask_final.transpose((1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = weight_masked.transpose((3, 2, 1, 0))
        elif len(weight_masked.shape) == 2:
            weight_masked = weight_masked.transpose((1, 0))

        return mask_final, weight_masked

    def random_prune(self, weight, prune_ratio, norm='l1'):
        """
            the random pruning method
        :param weight: tensor of the weight from the model: (kernel, kernel, in-channel, out-channel)
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        if len(weight.shape) == 4:
            weight = tf.transpose(weight, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = tf.transpose(weight, (1, 0))

        shape = weight.shape
        print("random pruning {}: {}".format(norm, weight.shape))
        weight2d = tf.reshape(weight, shape=(shape[0], -1))
        if norm == 'l1':
            weight_norm = tf.abs(weight2d)
        elif norm == 'l2':
            weight_norm = tf.square(weight2d)

        # find the threshold
        weight_vec = tf.reshape(weight_norm, (-1, 1))
        threshold = tf.gather(tf.contrib.framework.sort(weight_vec, axis=0), int(int(weight_vec.shape[0]) * prune_ratio))

        # set those > threshold as 1, those < threshold as 0
        zeros = tf.zeros(shape=weight_norm.shape, dtype='float32')
        ones = tf.ones(shape=weight_norm.shape, dtype='float32')
        mask_final = tf.where(tf.less_equal(weight_norm, threshold), zeros, ones)
        weight_masked = tf.multiply(weight2d, mask_final)
        weight_masked = tf.reshape(weight_masked, shape)
        mask_final = tf.reshape(mask_final, shape)

        if len(mask_final.shape) == 4:
            mask_final = tf.transpose(mask_final, (3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = tf.transpose(mask_final, (1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = tf.transpose(weight_masked, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight_masked = tf.transpose(weight_masked, (1, 0))

        return mask_final, weight_masked

    def random_prune_npy(self, weight, prune_ratio, norm='l1'):
        """
            the random pruning method
        :param weight: numpy.array of the weight from the model
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        percent = prune_ratio * 100

        if len(weight.shape) == 4:
            weight = weight.transpose((3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = weight.transpose((1, 0))

        shape = weight.shape
        print("random pruning {}: {}".format(norm, weight.shape))
        weight2d = weight.reshape(shape[0], -1)
        if norm == 'l1':
            weight_norm = np.abs(weight2d)
        elif norm == 'l2':
            weight_norm = np.square(weight2d)

        # find the threshold
        threshold = np.percentile(weight_norm, percent)
        # set those > threshold as 1, those < threshold as 0
        mask_final = np.where(weight_norm <= threshold, 0, 1)
        weight_masked = weight2d * mask_final
        weight_masked = weight_masked.reshape(shape)
        mask_final = mask_final.reshape(shape)

        if len(mask_final.shape) == 4:
            mask_final = mask_final.transpose((3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = mask_final.transpose((1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = weight_masked.transpose((3, 2, 1, 0))
        elif len(weight_masked.shape) == 2:
            weight_masked = weight_masked.transpose((1, 0))

        return mask_final, weight_masked

    def column_prune(self, weight, prune_ratio, norm='l1'):
        """
            the column pruning method
        :param weight: tensor of the weight from the model: (kernel, kernel, in-channel, out-channel)
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        if len(weight.shape) == 4:
            weight = tf.transpose(weight, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = tf.transpose(weight, (1, 0))

        shape = weight.shape
        print("column pruning {}: {}".format(norm, weight.shape))
        weight2d = tf.reshape(weight, shape=(shape[0], -1))
        if norm == 'l1':
            weight_norm = tf.abs(weight2d)
        elif norm == 'l2':
            weight_norm = tf.square(weight2d)

        # find the threshold
        weight_vec = tf.reduce_sum(weight_norm, axis=0)
        threshold = tf.gather(tf.contrib.framework.sort(weight_vec, axis=0), int(int(weight_vec.shape[0]) * prune_ratio))

        # set those > threshold as 1, those < threshold as 0
        zeros = tf.zeros(shape=weight_vec.shape, dtype='float32')
        ones = tf.ones(shape=weight_vec.shape, dtype='float32')
        mask_vec = tf.where(tf.less_equal(weight_vec, threshold), zeros, ones)
        mask_final = tf.ones([weight2d.shape[0], 1], dtype='float32') * mask_vec
        weight_masked = tf.multiply(weight2d, mask_final)
        weight_masked = tf.reshape(weight_masked, shape)
        mask_final = tf.reshape(mask_final, shape)

        if len(mask_final.shape) == 4:
            mask_final = tf.transpose(mask_final, (3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = tf.transpose(mask_final, (1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = tf.transpose(weight_masked, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight_masked = tf.transpose(weight_masked, (1, 0))

        return mask_final, weight_masked

    def column_prune_npy(self, weight, prune_ratio, norm='l1'):
        """
            the column pruning method
        :param weight: numpy.array of the weight from the model
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        percent = prune_ratio * 100

        if len(weight.shape) == 4:
            weight = weight.transpose((3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = weight.transpose((1, 0))

        shape = weight.shape
        print("column pruning {}: {}".format(norm, weight.shape))
        weight2d = weight.reshape(shape[0], -1)
        if norm == 'l1':
            weight_norm = np.abs(weight2d)
        elif norm == 'l2':
            weight_norm = np.square(weight2d)

        vec_norm = np.sum(weight_norm, axis=0)
        threshold = np.percentile(vec_norm, percent)
        under_threshold = vec_norm <= threshold
        above_threshold = vec_norm > threshold
        weight2d[:, under_threshold] = 0
        weight_masked = weight2d.reshape(shape)
        # mask
        mask = np.zeros(weight2d.shape, dtype=np.float32)
        mask[:, above_threshold] = 1
        mask_final = mask.reshape(shape)

        if len(mask_final.shape) == 4:
            mask_final = mask_final.transpose((3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = mask_final.transpose((1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = weight_masked.transpose((3, 2, 1, 0))
        elif len(weight_masked.shape) == 2:
            weight_masked = weight_masked.transpose((1, 0))

        return mask_final, weight_masked

    def filter_prune(self, weight, prune_ratio, norm='l1'):
        """
            the filter pruning method
        :param weight: tensor of the weight from the model: (kernel, kernel, in-channel, out-channel)
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        if len(weight.shape) == 4:
            weight = tf.transpose(weight, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = tf.transpose(weight, (1, 0))

        shape = weight.shape
        print("column pruning {}: {}".format(norm, weight.shape))
        weight2d = tf.reshape(weight, shape=(shape[0], -1))
        if norm == 'l1':
            weight_norm = tf.abs(weight2d)
        elif norm == 'l2':
            weight_norm = tf.square(weight2d)

        # find the threshold
        weight_vec = tf.reduce_sum(weight_norm, axis=1)
        threshold = tf.gather(tf.contrib.framework.sort(weight_vec, axis=0), int(int(weight_vec.shape[0]) * prune_ratio))

        # set those > threshold as 1, those < threshold as 0
        zeros = tf.zeros(shape=weight_vec.shape, dtype='float32')
        ones = tf.ones(shape=weight_vec.shape, dtype='float32')
        mask_vec = tf.where(tf.less_equal(weight_vec, threshold), zeros, ones)
        mask_final = mask_vec * tf.ones([1, weight2d.shape[1]], dtype='float32')
        weight_masked = tf.multiply(weight2d, mask_final)
        weight_masked = tf.reshape(weight_masked, shape)
        mask_final = tf.reshape(mask_final, shape)

        if len(mask_final.shape) == 4:
            mask_final = tf.transpose(mask_final, (3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = tf.transpose(mask_final, (1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = tf.transpose(weight_masked, (3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight_masked = tf.transpose(weight_masked, (1, 0))

        return mask_final, weight_masked

    def filter_prune_npy(self, weight, prune_ratio, norm='l1'):
        """
            the filter pruning method
        :param weight: numpy.array of the weight from the model
        :param prune_ratio: float value, 0.00 - 1.00
        :param norm: the normalization way
        :return:
            mask_final: the mask of the weight, 1 for save, 0 for prune
            weight_masked: masked weight, 0 for no value
        """
        percent = prune_ratio * 100

        if len(weight.shape) == 4:
            weight = weight.transpose((3, 2, 1, 0))
        elif len(weight.shape) == 2:
            weight = weight.transpose((1, 0))

        shape = weight.shape
        print("filter pruning {}: {}".format(norm, weight.shape))
        weight2d = weight.reshape(shape[0], -1)
        if norm == 'l1':
            weight_norm = np.abs(weight2d)
        elif norm == 'l2':
            weight_norm = np.square(weight2d)

        vec_norm = np.sum(weight_norm, axis=1)
        threshold = np.percentile(vec_norm, percent)
        under_threshold = vec_norm <= threshold
        above_threshold = vec_norm > threshold
        weight2d[under_threshold, :] = 0
        weight_masked = weight2d.reshape(shape)
        # mask
        mask = np.zeros(weight2d.shape, dtype=np.float32)
        mask[above_threshold, :] = 1
        mask_final = mask.reshape(shape)

        if len(mask_final.shape) == 4:
            mask_final = mask_final.transpose((3, 2, 1, 0))
        elif len(mask_final.shape) == 2:
            mask_final = mask_final.transpose((1, 0))

        if len(weight_masked.shape) == 4:
            weight_masked = weight_masked.transpose((3, 2, 1, 0))
        elif len(weight_masked.shape) == 2:
            weight_masked = weight_masked.transpose((1, 0))

        return mask_final, weight_masked
