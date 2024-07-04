# -*- coding: UTF-8 -*-
import unittest
import cv2
import torch
import numpy as np
from sys import version_info
# test expr
import MNN
import MNN.expr as expr
import MNN.cv as cv
import MNN.numpy as mp
import math

# numpy version
NUMPY_V1 = np.__version__ >= '1.0.0' and np.__version__ < '2.0.0'
NUMPY_V2 = np.__version__ >= '2.0.0' and np.__version__ < '3.0.0'

img_path = '../../resource/images/cat.jpg'

class UnitTest(unittest.TestCase):
    currentResult = None
    @classmethod
    def setUpClass(self):
        self.x_ = np.arange(0.001, 0.999, 0.000974609375).reshape(1, 4, 16, 16).astype(np.float32)
        self.x  = expr.const(self.x_, self.x_.shape)
        self._x = torch.Tensor(self.x_)
        self.img = cv.imread(img_path)
        self.img_ = cv2.imread(img_path)
        self.imgf_ = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        self.imgf = expr.const(self.imgf_, [1, self.imgf_.shape[0], self.imgf_.shape[1], 1], expr.NHWC)
    @classmethod
    def setResult(cls, amount, errors, failures, skipped):
        cls.amount, cls.errors, cls.failures, cls.skipped = amount, errors, failures, skipped
    def tearDown(self):
        amount = self.currentResult.testsRun
        errors = self.currentResult.errors
        failures = self.currentResult.failures
        skipped = self.currentResult.skipped
        self.setResult(amount, errors, failures, skipped)
    @classmethod
    def tearDownClass(cls):
        blocked = len(cls.errors)
        failed  = len(cls.failures)
        passed  = cls.amount - blocked - failed
        skipped = len(cls.skipped)
        try:
            print('\nTEST_NAME_PYMNN_UNIT: Pymnn单元测试\nTEST_CASE_AMOUNT_PYMNN_UNIT: {\"blocked\":%d,\"failed\":%d,\"passed\":%d,\"skipped\":%d}\n'%(blocked, failed, passed, skipped))
        except:
            print('\nTEST_NAME_PYMNN_UNIT: PymnnUnitTest\nTEST_CASE_AMOUNT_PYMNN_UNIT: {\"blocked\":%d,\"failed\":%d,\"passed\":%d,\"skipped\":%d}\n'%(blocked, failed, passed, skipped))
    def run(self, result=None):
        self.currentResult = result
        unittest.TestCase.run(self, result)
    def assertEqualArray(self, l, r, rtol=1e-3, atol=1e-3):
        self.assertEqual(l.shape, r.shape)
        self.assertTrue(np.allclose(l, r, rtol=rtol, atol=atol))
    def assertEqualVar(self, l, r, rtol=1e-3, atol=1e-3):
        # TODO: MNN not support empty Var
        if r.size == 0:
            return
        self.assertEqualArray(l.read().copy(), r, rtol, atol)
    def assertEqualVars(self, l, r, rtol=1e-3, atol=1e-3):
        for x,y in zip(l, r):
            self.assertEqualVar(x, y, rtol, atol)
    def assertEqualImg(self, l, r, rtol=1e-1, atol=2):
        self.assertEqualVar(expr.squeeze(l), r, rtol, atol)
    def assertEqualShape(self, l, r):
        self.assertEqual(len(l), len(r))
        for x,y in zip(l, r):
            self.assertEqual(x, y)
    def assertEqualPoints(self, l, r):
        l = np.asarray(l).astype(r.dtype).flatten()
        r = r.flatten()
        self.assertEqualArray(l, r)
    # test V2 api
    # test ImageProcess
    def test_Tensor(self):
        x = MNN.Tensor((2, 2), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), np.zeros([2, 2], dtype=np.float32))
        data = np.array([[1., 2.], [3., 4.]], dtype=np.float32);
        x = MNN.Tensor((2, 2), MNN.Halide_Type_Float, [1., 2., 3., 4.], MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, (1., 2., 3., 4.), MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, data.tobytes(), MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, data.__array_interface__['data'][0], MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, mp.array([[1., 2.], [3., 4.]]).ptr, MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualArray(x.getNumpyData(), data)
    def test_image_process(self):
        src = np.asarray([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
        dst = np.asarray([[10, 100], [200, 20], [100, 250]], dtype=np.float32)
        ih, iw, ic = self.img_.shape
        # test tensor create
        tmp = MNN.Tensor((1, ih, iw, ic), MNN.Halide_Type_Uint8, MNN.Tensor_DimensionType_Tensorflow)
        dest = MNN.Tensor(tmp, MNN.Tensor_DimensionType_Tensorflow)
        # 1. create config
        config = {
            'filterType': MNN.CV_Filter_BILINEAL,
            'sourceFormat': MNN.CV_ImageFormat_BGR,
            'destFormat': MNN.CV_ImageFormat_BGR,
            'wrap': MNN.CV_Wrap_ZERO,
            'mean': (0., 0., 0., 0.),
            'normal': (1., 1., 1., 1.),
        }
        # 2. create ImageProcess
        proc = MNN.CVImageProcess(config)
        # 3. create Matrix
        m = MNN.CVMatrix()
        m.setPolyToPoly(src, dst)
        # 4. set Matrix and padding
        proc.setPadding(100)
        proc.setMatrix(m)
        # get data pointer of ndarray
        ptr, _ = self.img_.__array_interface__['data']
        proc.convert(ptr, iw, ih, iw * ic, dest)
        dest = dest.getNumpyData()
        dest = np.squeeze(dest, 0)
        # opencv like
        n = cv2.getAffineTransform(src, dst)
        n = cv2.invertAffineTransform(n)
        dest_ = cv2.warpAffine(self.img_, n, (iw, ih), borderValue=(100, 100, 100))
        # cv2.imwrite('cv.jpg', dest_)
        # cv2.imwrite('mnn.jpg', dest)
        # img is like
        self.assertEqual(dest.shape, dest_.shape)
    # test unary
    def test_sign(self):
        self.assertEqualVar(expr.sign(self.x), np.sign(self.x_))
    def test_abs(self):
        self.assertEqualVar(expr.abs(self.x), np.abs(self.x_))
    def test_negative(self):
        self.assertEqualVar(expr.negative(self.x), np.negative(self.x_))
    def test_floor(self):
        self.assertEqualVar(expr.floor(self.x), np.floor(self.x_))
    def test_ceil(self):
        self.assertEqualVar(expr.ceil(self.x), np.ceil(self.x_))
    def test_square(self):
        self.assertEqualVar(expr.square(self.x), np.square(self.x_))
    def test_sqrt(self):
        self.assertEqualVar(expr.sqrt(self.x), np.sqrt(self.x_))
    def test_rsqrt(self):
        self.assertEqualVar(expr.rsqrt(self.x), 1 / np.sqrt(self.x_))
    def test_exp(self):
        self.assertEqualVar(expr.exp(self.x), np.exp(self.x_))
    def test_log(self):
        self.assertEqualVar(expr.log(self.x), np.log(self.x_))
    def test_sin(self):
        self.assertEqualVar(expr.sin(self.x), np.sin(self.x_))
    def test_cos(self):
        self.assertEqualVar(expr.cos(self.x), np.cos(self.x_))
    def test_tan(self):
        self.assertEqualVar(expr.tan(self.x), np.tan(self.x_))
    def test_asin(self):
        self.assertEqualVar(expr.asin(self.x), np.arcsin(self.x_))
    def test_acos(self):
        self.assertEqualVar(expr.acos(self.x), np.arccos(self.x_))
    def test_atan(self):
        self.assertEqualVar(expr.atan(self.x), np.arctan(self.x_))
    def test_reciprocal(self):
        self.assertEqualVar(expr.reciprocal(self.x), np.reciprocal(self.x_))
    def test_log1p(self):
        self.assertEqualVar(expr.log1p(self.x + 0.1), np.log1p(self.x_ + 0.1))
    def test_tanh(self):
        self.assertEqualVar(expr.tanh(self.x), torch.tanh(self._x))
    def test_sigmoid(self):
        self.assertEqualVar(expr.sigmoid(self.x), 1/(1+np.exp(-self.x_)))
    # test binary
    def test_add(self):
        self.assertEqualVar(expr.add(self.x, self.x), np.add(self.x_, self.x_))
        self.assertEqualVar(self.x + self.x, self.x_ + self.x_)
    def test_subtract(self):
        self.assertEqualVar(expr.subtract(self.x, self.x), np.subtract(self.x_, self.x_))
        self.assertEqualVar(self.x - self.x, self.x_ - self.x_)
    def test_multiply(self):
        self.assertEqualVar(expr.multiply(self.x, self.x), np.multiply(self.x_, self.x_))
        self.assertEqualVar(self.x * self.x, self.x_ * self.x_)
    def test_divide(self):
        self.assertEqualVar(expr.divide(self.x, self.x), np.divide(self.x_, self.x_))
        self.assertEqualVar(self.x / self.x, self.x_ / self.x_)
    def test_pow(self):
        self.assertEqualVar(expr.pow(self.x, self.x), np.power(self.x_, self.x_))
        self.assertEqualVar(self.x ** self.x, self.x_ ** self.x_)
    def test_minimum(self):
        self.assertEqualVar(expr.minimum(self.x, self.x), np.minimum(self.x_, self.x_))
    def test_maximum(self):
        self.assertEqualVar(expr.maximum(self.x, self.x), np.maximum(self.x_, self.x_))
    def test_bias_add(self):
        bias_ = np.random.randn(16).astype(np.float32)
        bias = expr.const(bias_, bias_.shape)
        self.assertEqualVar(expr.bias_add(self.x, bias), np.add(self.x_, bias_))
    def test_greater(self):
        self.assertEqualVar(expr.greater(self.x, self.x), np.greater(self.x_, self.x_))
    def test_greater_equal(self):
        self.assertEqualVar(expr.greater_equal(self.x, self.x), np.greater_equal(self.x_, self.x_))
    def test_less(self):
        self.assertEqualVar(expr.less(self.x, self.x), np.less(self.x_, self.x_))
    def test_floordiv(self):
        self.assertEqualVar(expr.floordiv(2.0, 1.2), np.floor_divide(2.0, 1.2))
    def test_less(self):
        self.assertEqualVar(expr.less(self.x, self.x), np.less(self.x_, self.x_))
    def test_squared_difference(self):
        self.assertEqualVar(expr.squared_difference(self.x, self.x), np.square(self.x_ - self.x_))
    def test_equal(self):
        self.assertEqualVar(expr.equal(self.x, self.x), np.equal(self.x_, self.x_))
    def test_not_equal(self):
        self.assertEqualVar(expr.not_equal(self.x, self.x), np.not_equal(self.x_, self.x_))
    def test_less_equal(self):
        self.assertEqualVar(expr.less_equal(self.x, self.x), np.less_equal(self.x_, self.x_))
    def test_floormod(self):
        self.assertEqualVar(expr.floormod(self.x, self.x + 0.1), self.x_ - (np.floor(self.x_ / (self.x_+0.1)) * (self.x_+0.1)))
    # test reduce
    def test_reduce_sum(self):
        self.assertEqualVar(expr.reduce_sum(self.x), np.sum(self.x_))
    def test_reduce_mean(self):
        self.assertEqualVar(expr.reduce_mean(self.x), np.mean(self.x_))
    def test_reduce_max(self):
        self.assertEqualVar(expr.reduce_max(self.x), np.max(self.x_))
    def test_reduce_min(self):
        self.assertEqualVar(expr.reduce_min(self.x), np.min(self.x_))
    def test_reduce_prod(self):
        self.assertEqualVar(expr.reduce_prod(self.x), np.prod(self.x_))
    def test_reduce_any(self):
        x = expr.const([1, -2], [2], expr.NCHW, expr.int)
        self.assertEqual(expr.reduce_any(x).read_as_tuple(), (int(np.any(x.read().copy())),))
    def test_reduce_all(self):
        x = expr.const([1, -2], [2], expr.NCHW, expr.int)
        self.assertEqual(expr.reduce_all(x).read_as_tuple(), (int(np.all(x.read().copy())),))
    # test eltwise
    def test_eltwise_prod(self):
        self.assertEqualVar(expr.eltwise_prod(self.x, self.x, []), np.multiply(self.x_, self.x_))
    def test_eltwise_sum(self):
        self.assertEqualVar(expr.eltwise_sum(self.x, self.x, []), np.add(self.x_, self.x_))
    def test_eltwise_max(self):
        self.assertEqualVar(expr.eltwise_max(self.x, self.x, []), np.maximum(self.x_, self.x_))
    def test_eltwise_sub(self):
        self.assertEqualVar(expr.eltwise_sub(self.x, self.x, []), np.subtract(self.x_, self.x_))
    # test nn
    def test_cast(self):
        self.assertEqualVar(expr.cast(self.x, expr.int), self.x_.astype(np.int32))
    def test_matmul(self):
        self.assertEqualVar(expr.matmul(self.x, self.x), np.matmul(self.x_, self.x_))
    def test_normalize(self):
        def _refNormalize(src, batch, channel, area, scale, eps):
            dst = [0.0] * (batch * channel * area)
            for b in range(0, batch):
                for x in range(0, area):
                    dstX = b * area * channel + x;
                    srcX = b * area * channel + x;
                    sumSquare = 0.0;
                    for c in range(0, channel):
                        sumSquare += src[srcX + area * c] * src[srcX + area * c];
                    normalValue = 1.0 / math.sqrt(sumSquare + eps);
                    for c in range(0, channel):
                        dst[dstX + area*c] = src[srcX + area * c] * normalValue * scale[c];
            return dst
        src = [-1.0, -2.0, 3.0, 4.0]
        dst = _refNormalize(src, 1, 2, 2, [0.5, 0.5], 0.0)
        x = expr.const(src, [1, 2, 2, 1], expr.NCHW)
        y = expr.const(dst, [1, 2, 2, 1], expr.NCHW)
        self.assertEqualVar(expr.normalize(x, 0, 0, 0.0, [0.5, 0.5]), y.read().copy())
    def test_argmax(self):
        x = expr.reshape(self.x, [-1])
        x_ = np.reshape(self.x_, [-1])
        self.assertEqualVar(expr.argmax(x), np.argmax(x_))
    def test_unravel_index(self):
        indice = expr.const([22, 41, 37], [3], expr.NCHW, expr.int)
        shape  = expr.const([7, 6], [2], expr.NCHW, expr.int)
        npres = []
        for x in [22, 41, 37]:
            npres.append(list(np.unravel_index(x, [7, 6])))
        npres = np.asarray(npres).transpose([1, 0])
        self.assertEqualVar(expr.unravel_index(indice, shape), npres)
    def test_scatter_nd(self):
        indices = expr.const([4, 3, 1, 7], [4, 1], expr.NHWC, expr.int)
        updates = expr.const([9.0, 10.0, 11.0, 12.0], [4], expr.NHWC, expr.float)
        shape = expr.const([8], [1], expr.NHWC, expr.int)
        self.assertEqual(expr.scatter_nd(indices, updates, shape).read_as_tuple(), (0.0, 11, 0, 10, 9, 0, 0, 12))
    def test_one_hot(self):
        indices = expr.const([0, 1, 2], [3], expr.NHWC, expr.int)
        self.assertEqual(expr.one_hot(indices, 3).read_as_tuple(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    def test_broadcast_to(self):
        self.assertEqualVar(expr.broadcast_to(self.x, [2, 4, 16, 16]),
                              np.broadcast_to(self.x_, [2, 4, 16, 16]))
    def test_placeholder(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = expr.placeholder([2, 3], expr.NCHW, expr.float)
        x.write(vals)
        self.assertEqualVar(x, np.asarray(vals).reshape([2,3]))
    def test_clone(self):
        self.assertEqualVar(expr.clone(self.x), np.copy(self.x_))
    def test_const(self):
        self.assertEqualVar(self.x, self.x_)
        list_data = [1., 2., 3., 4.]
        tuple_data = (1., 2., 3., 4.)
        data = np.array(list_data, dtype=np.float32).reshape([2, 2])
        self.assertEqualVar(expr.const(list_data, [2, 2]), data)
        self.assertEqualVar(expr.const(tuple_data, [2, 2]), data)
        self.assertEqualVar(expr.const(data, [2, 2]), data)
        self.assertEqualVar(expr.const(data.tobytes(), [2, 2]), data)
        self.assertEqualVar(expr.const(data.__array_interface__['data'][0], [2, 2]), data)
        x = MNN.Tensor([2, 2], MNN.Halide_Type_Float, (1., 2., 3., 4.), MNN.Tensor_DimensionType_Tensorflow)
        self.assertEqualVar(expr.const(x.getHost(), [2, 2]), data)
    def test_conv2d(self):
        w_ = np.random.randn(2, 4, 3, 3).astype(np.float32)
        b_ = np.random.randn(2).astype(np.float32)
        w = expr.const(w_, w_.shape)
        b = expr.const(b_, b_.shape)
        _w = torch.Tensor(w_)
        _b = torch.Tensor(b_)
        x = expr.convert(self.x, expr.NC4HW4)
        self.assertEqualVar(expr.convert(expr.conv2d(x, w, b), expr.NCHW),
                              torch.conv2d(self._x, _w, _b))
    def test_conv2d_transpose(self):
        w_ = np.random.randn(4, 2, 3, 3).astype(np.float32)
        b_ = np.random.randn(2).astype(np.float32)
        w = expr.const(w_, w_.shape)
        b = expr.const(b_, b_.shape)
        _w = torch.Tensor(w_)
        _b = torch.Tensor(b_)
        x = expr.convert(self.x, expr.NC4HW4)
        self.assertEqualVar(expr.convert(expr.conv2d_transpose(x, w, b), expr.NCHW),
                              torch.conv_transpose2d(self._x, _w, _b))
    def test_max_pool(self):
        m = torch.nn.MaxPool2d([3,3], [2,2])
        self.assertEqualVar(expr.max_pool(self.x, [3,3], [2,2]), m(self._x))
    def test_avg_pool(self):
        m = torch.nn.AvgPool2d([3,3], [2,2])
        self.assertEqualVar(expr.avg_pool(self.x, [3,3], [2,2]), m(self._x))
    def test_reshape(self):
        self.assertEqualVar(expr.reshape(self.x, [16, -1]), np.reshape(self.x_, [16, -1]))
    def test_scale(self):
        x = expr.const([-1.0, -2.0, 3.0, 4.0], [1, 2, 2, 1], expr.NCHW, expr.float)
        x = expr.convert(x, expr.NC4HW4)
        y = expr.scale(x, 4, [2.0, 1.0], [3.0, 4.0])
        y = expr.convert(y, expr.NCHW)
        self.assertEqual(y.read_as_tuple(), (1, -1, 7, 8))
    def test_relu(self):
        self.assertEqualVar(expr.relu(self.x), torch.relu(self._x))
    def test_relu6(self):
        m = torch.nn.ReLU6()
        self.assertEqualVar(expr.relu6(self.x), m(self._x))
    def test_prelu(self):
        x = expr.convert(self.x, expr.NC4HW4)
        x = expr.prelu(x, [0.5, 0.5, 0.5, 0.5])
        x = expr.convert(x, expr.NCHW)
        y = torch.prelu(self._x, torch.Tensor([0.5, 0.5, 0.5, 0.5]))
        self.assertEqualVar(x, y)
    def test_softmax(self):
        self.assertEqualVar(expr.softmax(self.x, 1), torch.softmax(self._x, 1))
    def test_softplus(self):
        self.assertEqualVar(expr.softplus(self.x), np.log(np.add(np.exp(self.x_), 1)))
    def test_softsign(self):
        self.assertEqualVar(expr.softsign(self.x), np.divide(self.x_, np.add(np.abs(self.x_), 1)))
    def test_slice(self):
        start = expr.const([0, 0, 4, 4], [4], expr.NCHW, expr.int)
        size  = expr.const([1, 2, 4, 4], [4], expr.NCHW, expr.int)
        self.assertEqualVar(expr.slice(self.x, start, size), self._x[0:1,0:2,4:8,4:8])
    def test_split(self):
        mnnres = expr.split(self.x, [2, 2], 1)
        torchres = torch.split(self._x, [2, 2], 1)
        for x,y in zip(mnnres, torchres):
            self.assertEqualVar(x, y)
    def test_strided_slice(self):
        begin  = expr.const([0, 1, 4, 4], [4], expr.NCHW, expr.int)
        end    = expr.const([1, 3, 16, 16], [4], expr.NCHW, expr.int)
        stride = expr.const([1, 1, 2, 2], [4], expr.NCHW, expr.int)
        self.assertEqualVar(expr.strided_slice(self.x, begin, end, stride, 0, 0, 0, 0, 0),
                              self._x[0:1:1,1:3:1,4:16:2,4:16:2])
    def test_concat(self):
        self.assertEqualVar(expr.concat([self.x, self.x], 0), torch.cat([self._x, self._x], 0))
    def test_convert(self):
        nc4hw4_x = expr.convert(self.x, expr.NC4HW4)
        self.assertEqual(nc4hw4_x.data_format, expr.NC4HW4)
    def test_transpose(self):
        self.assertEqualVar(expr.transpose(self.x, [0, 2, 3, 1]), np.transpose(self.x_, [0, 2, 3, 1]))
    def test_channel_shuffle(self):
        x = expr.const(np.arange(8).astype(np.float32), [1, 1, 2, 4], expr.NHWC, expr.float)
        y = expr.convert(expr.channel_shuffle(x, 2), expr.NHWC).read_as_tuple()
        self.assertEqual(y, (0, 2, 1, 3, 4, 6, 5, 7,))
    @unittest.skip("skip for this case, wrong sometimes")
    def test_reverse_sequence(self):
        vals = []
        for o in range(6):
            for i in range(4):
                for m in range(7):
                    for j in range(10):
                        for k in range(8):
                            vals.append(float(10000 * o + 1000 * i + 100 * m + 10 * j + k))
        seq = [7, 2, 3, 5]
        y = expr.const(seq, [4], expr.NHWC, expr.int)
        x = expr.const(vals, [6, 4, 7, 10, 8], expr.NHWC, expr.float)
        z = expr.reverse_sequence(x, y, 1, 3).read().copy()
        for o in range(6):
            for i in range(4):
                req = seq[i]
                for m in range(7):
                    for j in range(10):
                        for k in range(8):
                            if j < req:
                                need = 10000 * o + 1000 * i + 100 * m + 10 * (req - j - 1) + k
                            else:
                                need = 10000 * o + 1000 * i + 100 * m + 10 * j + k
                            delta = abs(z[o, i, m, j, k] - float(need))
                            if delta > 1e-4:
                                self.assertTrue(False)
    def test_crop(self):
        vals = np.arange(1.0, 17.0).astype(np.float32)
        x = expr.const(vals, [1, 1, 4, 4], expr.NCHW, expr.float)
        x = expr.convert(x, expr.NC4HW4)
        size = expr.const([0.0, 0.0, 0.0, 0.0], [1, 1, 2, 2], expr.NCHW, expr.float)
        self.assertEqual(expr.convert(expr.crop(x, size, 2, [1, 1]), expr.NCHW).read_as_tuple(), (6.0, 7.0, 10.0, 11.0))
    def test_resize(self):
        x = expr.const([-1.0, -2.0, 3.0, 4.0], [1, 2, 2, 1], expr.NHWC, expr.float)
        x = expr.convert(x, expr.NC4HW4)
        y = expr.resize(x, 2.0, 2.0)
        y = expr.convert(y, expr.NHWC)
        z = np.asarray([-1.0, -1.5, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0,
                        3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0]).reshape([1, 4, 4, 1]).astype(np.float32)
        self.assertEqualVar(y, z)
    def test_pad(self):
        m = torch.nn.ConstantPad2d([2, 2, 1, 1], 0)
        pad = expr.const([1, 1, 2, 2], [2, 2], expr.NCHW, expr.int)
        x = expr.reshape(self.x, [16, 64])
        _x = torch.reshape(self._x, [16, 64])
        self.assertEqualVar(expr.pad(x, pad, expr.CONSTANT), m(_x))
    def test_shape(self):
        self.assertEqual(self.x.shape, list(self.x_.shape))
    def test_stack(self):
        self.assertEqualVar(expr.stack([self.x, self.x], 0), torch.stack([self._x, self._x], 0))
    def test_fill(self):
        dims = expr.const([3, 4, 4], [3], expr.NCHW, expr.int)
        value = expr.scalar(7.0)
        self.assertEqualVar(expr.fill(dims, value), np.ones([3, 4, 4]) * 7)
    def test_tile(self):
        x = expr.const([-1.0, -2.0, 3.0, 4.0], [2,2])
        mul = expr.const([2, 2], [2], expr.NCHW, expr.int)
        # self.assertEqualVar(expr.tile(x, mul), torch.tile(torch.Tensor(x.read().copy()), [2, 2]))
        self.assertEqualVar(expr.tile(x, mul), np.tile(x.read().copy(), [2, 2]))
    def test_gather(self):
        vals = np.arange(1, 25).reshape(4, 3, 2).astype(np.float32)
        x  = expr.const(vals, vals.shape)
        indice = expr.const([1, 0, 1, 0], [4], expr.NCHW, expr.int)
        self.assertEqual(expr.gather(x, indice).read_as_tuple(), (7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    def test_select(self):
        cond = expr.const([1, 0, 1, 0], [4], expr.NCHW, expr.int)
        left = expr.const([1.0, 2.0, 3.0, 4.0], [4])
        right = expr.const([5.0, 6.0, 7.0, 8.0], [4])
        _cond = torch.Tensor(cond.read().copy()).to(torch.bool)
        _left = torch.Tensor(left.read().copy())
        _right = torch.Tensor(right.read().copy())
        self.assertEqualVar(expr.select(cond, left, right), torch.where(_cond, _left, _right))
    def test_squeeze(self):
        self.assertEqualVar(expr.squeeze(self.x, []), torch.squeeze(self._x))
    def test_unsqueeze(self):
        self.assertEqualVar(expr.unsqueeze(self.x, [0]), torch.unsqueeze(self._x, 0))
    def test_batch_to_space_nd(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        y = np.asarray(vals).reshape([1, 2, 2, 3]).astype(np.float32)
        x = expr.const(vals, [4, 1, 1, 3], expr.NHWC, expr.float)
        x = expr.convert(x, expr.NC4HW4)
        block_shape = expr.const([2, 2], [2], expr.NCHW, expr.int)
        crops = expr.const([0, 0, 0, 0], [2, 2], expr.NCHW, expr.int)
        self.assertEqualVar(expr.convert(expr.batch_to_space_nd(x, block_shape, crops), expr.NHWC), y)
    def test_gather_nd(self):
        parameter = expr.const([7.0, 2.0, 4.0, 6.0], [2, 2], expr.NHWC, expr.float)
        indice = expr.const([0, 0, 1, 1], [2, 2], expr.NHWC, expr.int)
        self.assertEqual(expr.gather_nd(parameter, indice).read_as_tuple(), (7.0, 6.0))
    def test_selu(self):
        self.assertEqualVar(expr.selu(self.x, 1.0507, 1.673263), torch.selu(self._x))
    def test_size(self):
        self.assertEqual(expr.size(self.x).read_as_tuple(), (self.x_.size,))
    def test_elu(self):
        m = torch.nn.ELU(0.5)
        self.assertEqualVar(expr.elu(self.x, 0.5), m(self._x))
    def test_matrix_band_part(self):
        matrix = expr.const([0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, -3.0, -2.0, -1.0, 0.0], [4, 4], expr.NHWC, expr.float)
        lower  = expr.scalar(1)
        upper  = expr.scalar(-1)
        y = np.asarray([0, 1, 2, 3, -1, 0, 1, 2, -0, -1, 0, 1, -0, -0, -1, 0]).reshape([4, 4]).astype(np.float32)
        self.assertEqualVar(expr.matrix_band_part(matrix, lower, upper), y)
    def test_moments(self):
        x = expr.const([0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, -3.0, -2.0, -1.0, 0.0], [1, 4, 4, 1], expr.NCHW, expr.float)
        x = expr.convert(x, expr.NC4HW4)
        shift = expr.scalar(1.0)
        res = expr.moments(x, [2, 3], shift, True)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].read_as_tuple(), (1.5, 0.5, -0.5, -1.5))   # mean
        self.assertEqual(res[1].read_as_tuple(), (1.25, 1.25, 1.25, 1.25)) # var
    def test_setdiff1d(self):
        x = expr.const([-1, 2, -3, 4, 5, -6, 7, -8, -9, -10, 11, 12, 13, 14, -15, -16], [16], expr.NHWC, expr.int)
        y = expr.const([-1, 2, -3, 4, 5, -6, 7, -8], [8], expr.NHWC, expr.int)
        self.assertEqual(expr.setdiff1d(x, y).read_as_tuple(), (-9, -10, 11, 12, 13, 14, -15, -16))
    def test_space_to_depth(self):
        x = expr.const([-1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 7.0, -8.0, -9.0, -10.0, 11.0, 12.0, 13.0, 14.0, -15.0, -16.0], [1, 4, 4, 1], expr.NHWC, expr.float)
        y = np.asarray([-1.0, 2.0, 5.0, 6.0, -3.0, 4.0, 7.0, -8.0, -9.0, -10.0, 13.0, 14.0, 11.0, 12.0, -15.0, -16.0]).reshape([1, 2, 2, 4]).astype(np.float32)
        self.assertEqualVar(expr.space_to_depth(x, 2), y)
    def test_zeros_like(self):
        self.assertEqualVar(expr.zeros_like(self.x), torch.zeros_like(self._x))
    def test_unstack(self):
        mnnres = expr.unstack(self.x, 1)
        torchres = torch.unbind(self._x, 1)
        for x,y in zip(mnnres, torchres):
            self.assertEqualVar(x, y)
    def test_rank(self):
        self.assertEqual(expr.rank(self.x).read_as_tuple(), (len(self.x_.shape),))
    def test_range(self):
        start = expr.const([0.0], [1])
        limit = expr.const([2.0], [1])
        delta = expr.const([0.3], [1])
        self.assertEqualVar(expr.range(start, limit, delta), np.arange(0.0, 2.0, 0.3))
    def test_depth_to_space(self):
        self.assertEqualVar(expr.depth_to_space(self.x, 2), torch.pixel_shuffle(self._x, 2))
    def test_sort(self):
        x = mp.array([5, -1, 2, 0])
        x_ = np.array([5, -1, 2, 0])
        self.assertEqualVar(expr.sort(x), np.sort(x_))
    def test_raster(self):
        x = mp.array([[1, 2], [3, 4]])
        x_ = np.array([[1, 2], [3, 4]])
        self.assertEqualVar(expr.raster([x], [0, 1, 1, 2, 0, 1, 2, 1, 1, 2, 2], [2, 2]), x_.transpose())
    def test_detection_post_process(self):
        pass
    # test cv
    # imgcodecs
    def test_haveImageReader(self):
        self.assertEqual(cv.haveImageReader(img_path), cv2.haveImageReader(img_path))
    def test_haveImageWriter(self):
        self.assertEqual(cv.haveImageWriter(img_path), cv2.haveImageWriter(img_path))
    def test_imdecode(self):
        buf = np.fromfile(img_path, dtype='uint8')
        x = cv.imdecode(buf, cv.IMREAD_COLOR)
        y = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertEqualImg(x, y)
    def test_imencode(self):
        a, x = cv.imencode('.png', self.img)
        b, y = cv2.imencode('.png', self.img_)
        self.assertEqual(a, b)
    def test_imread(self):
        self.assertEqualImg(self.img, self.img_)
        self.assertEqualImg(self.imgf, self.imgf_)
    def test_imwrite(self):
        cv2.imwrite('cv.jpg', self.img_)
        cv.imwrite('mnn.jpg', self.img)
        x = cv.imread('mnn.jpg')
        y = cv2.imread('cv.jpg')
        self.assertEqualImg(x, y, atol=18)
    # color
    def test_cvtColor(self):
        x = cv.cvtColor(self.img, cv.COLOR_RGB2BGR)
        y = cv2.cvtColor(self.img_, cv2.COLOR_RGB2BGR)
        self.assertEqualImg(x, y)
    @unittest.skip("mnn's YUV -> RGB is different from opencv, but it's right")
    def test_cvtColorTwoPlane(self):
        pass
    # filter
    def test_bilateralFilter(self):
        x = cv.bilateralFilter(self.img, 20, 80.0, 35.0)
        y = cv2.bilateralFilter(self.img_, 20, 80.0, 35.0)
        self.assertEqualImg(x, y)
    def test_blur(self):
        x = cv.blur(self.imgf, (3, 3))
        y = cv2.blur(self.imgf_, (3, 3))
        self.assertEqualImg(x, y)
    def test_boxFilter(self):
        x = cv.boxFilter(self.imgf, -1, (3, 3))
        y = cv2.boxFilter(self.imgf_, -1, (3, 3))
        self.assertEqualImg(x, y)
    def test_erode(self):
        x = cv.erode(self.imgf, cv.getStructuringElement(0, (3, 3)))
        y = cv2.erode(self.imgf_, cv2.getStructuringElement(0, (3, 3)))
        self.assertEqualImg(x, y)
    def test_dilate(self):
        x = cv.dilate(self.imgf, cv.getStructuringElement(0, (3, 3)))
        y = cv2.dilate(self.imgf_, cv2.getStructuringElement(0, (3, 3)))
        self.assertEqualImg(x, y)
    def test_filter2D(self):
        cvKernel = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        mnnKernel = expr.const(cvKernel, [3, 3])
        x = cv.filter2D(self.imgf, -1, mnnKernel)
        y = cv2.filter2D(self.imgf_, -1, cvKernel)
        self.assertEqualImg(x, y)
    def test_GaussianBlur(self):
        x = cv.GaussianBlur(self.imgf, (3, 3), 10)
        y = cv2.GaussianBlur(self.imgf_, (3, 3), 10)
        self.assertEqualImg(x, y)
    def test_getDerivKernels(self):
        x0, x1 = cv.getDerivKernels(1, 2, 1)
        y0, y1 = cv2.getDerivKernels(1, 2, 1)
        y0 = y0.reshape(x0.shape)
        y1 = y1.reshape(x1.shape)
        self.assertEqualVar(x0, y0)
        self.assertEqualVar(x1, y1)
    def test_getGaborKernel(self):
        x = cv.getGaborKernel((3, 3), 10, 5, 5, 5)
        y = cv2.getGaborKernel((3, 3), 10, 5, 5, 5)
        self.assertEqualVar(x, y)
    def test_getGaussianKernel(self):
        x = cv.getGaussianKernel(3, 5)
        y = cv2.getGaussianKernel(3, 5)
        y = y.reshape(x.shape)
        self.assertEqualVar(x, y)
    def test_getStructuringElement(self):
        x = cv.getStructuringElement(0, (3, 3))
        y = cv2.getStructuringElement(0, (3, 3))
        self.assertEqualVar(x, y)
    def test_Laplacian(self):
        x = cv.Laplacian(self.imgf, -1)
        y = cv2.Laplacian(self.imgf_, -1)
        self.assertEqualImg(x, y)
    def test_pyrDown(self):
        x = cv.pyrDown(self.imgf)
        y = cv2.pyrDown(self.imgf_)
        self.assertEqualImg(x, y)
    def test_pyrUp(self):
        x = cv.pyrUp(self.imgf)
        y = cv2.pyrUp(self.imgf_)
        self.assertEqualImg(x, y, 0.1, 11)
    def test_Scharr(self):
        x = cv.Scharr(self.imgf, -1, 1, 0)
        y = cv2.Scharr(self.imgf_, -1, 1, 0)
        self.assertEqualImg(x, y)
    def test_sepFilter2D(self):
        kernelX = np.asarray([[0, -1, 0]], dtype=np.float32)
        kernelY = np.asarray([[-1, 0, -1]], dtype=np.float32)
        mnnKernelX = expr.const(kernelX, (1, 3))
        mnnKernelY = expr.const(kernelY, (1, 3))
        x = cv.sepFilter2D(self.imgf, -1, mnnKernelX, mnnKernelY, 1)
        y = cv2.sepFilter2D(self.imgf_, -1, kernelX, kernelY, delta=1)
        self.assertEqualImg(x, y)
    def test_Sobel(self):
        x = cv.Sobel(self.imgf, -1, 1, 0)
        y = cv2.Sobel(self.imgf_, -1, 1, 0)
        self.assertEqualImg(x, y)
    def test_sqrBoxFilter(self):
        x = cv.sqrBoxFilter(self.imgf, -1, (1,1))
        y = cv2.sqrBoxFilter(self.imgf_, -1, (1,1))
        self.assertEqualImg(x, y)
    # geometric
    def test_getAffineTransform(self):
        src = np.asarray([[50, 50], [200, 50], [50, 200]], dtype=np.float32)
        dst = np.asarray([[10, 100], [200, 20], [100, 250]], dtype=np.float32)
        x = cv.getAffineTransform(src, dst)
        y = cv2.getAffineTransform(src, dst)
        x = np.asarray(x.read()[:6]).reshape(y.shape)
        self.assertEqualArray(x, y)
    def test_getPerspectiveTransform(self):
        src = np.asarray([[0, 0], [479, 0], [0, 359], [479, 359]], dtype=np.float32)
        dst = np.asarray([[0, 46.8], [432, 0], [96, 252], [384, 360]], dtype=np.float32)
        x = cv.getPerspectiveTransform(src, dst)
        y = cv2.getPerspectiveTransform(src, dst)
        x = np.asarray(x.read()).reshape(y.shape)
        self.assertEqualArray(x, y)
    def test_getRectSubPix(self):
        x = cv.getRectSubPix(self.img, (11, 11), (10.0, 10.0))
        y = cv2.getRectSubPix(self.img_, (11, 11), (10, 10))
        self.assertEqualImg(x, y)
    def test_getRotationMatrix2D(self):
        x = cv.getRotationMatrix2D((10., 10.), 50, 0.6)
        y = cv2.getRotationMatrix2D((10., 10.), 50, 0.6)
        x = np.asarray(x.read()[:6]).reshape(y.shape)
        self.assertEqualArray(x, y)
    def test_invertAffineTransform(self):
        cvM = np.asarray([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        mnnM = MNN.CVMatrix()
        mnnM.write(cvM)
        x = cv.invertAffineTransform(mnnM)
        y = cv2.invertAffineTransform(cvM)
        x = np.asarray(x.read()[:6]).reshape(y.shape)
        self.assertEqualArray(x, y)
    def test_remap(self):
        row, col, ch = self.img_.shape
        mapx_ = np.ones(self.img_.shape[:2], np.float32)
        mapy_ = np.ones(self.img_.shape[:2], np.float32)
        for i in range(row):
            for j in range(col):
                if NUMPY_V1:
                    mapx_.itemset((i, j), j)
                    mapy_.itemset((i, j), row-i)
                elif NUMPY_V2:
                    mapx_[i, j] = j
                    mapy_[i, j] = row - i
        mapx = expr.const(mapx_, mapx_.shape)
        mapy = expr.const(mapy_, mapy_.shape)
        x = cv.remap(self.img, mapx, mapy, cv.INTER_LINEAR)
        y = cv2.remap(self.img_, mapx_, mapy_, cv2.INTER_LINEAR)
        self.assertEqualImg(x, y)
    def test_resize(self):
        x = cv.resize(self.img, (180, 240))
        y = cv2.resize(self.img_, (180, 240))
        self.assertEqualImg(x, y)
    def test_warpAffine(self):
        cvM = np.asarray([[0.5, 0, 0], [0, 0.8, 0]], dtype=np.float32)
        mnnM = MNN.CVMatrix()
        mnnM.write(cvM)
        x = cv.warpAffine(self.img, mnnM, (480, 360))
        y = cv2.warpAffine(self.img_, cvM, (480, 360))
        self.assertEqualImg(x, y)
    @unittest.skip("skip for this case is wrong now")
    def test_warpPerspective(self):
        cvM = np.asarray([[0.40369818, 0.37649557, 0],
                          [-0.097703546, 0.85793871, 46.799999],
                          [-0.0011531961, 0.0011363134, 1]], dtype=np.float32)
        mnnM = MNN.CVMatrix()
        mnnM.write(cvM)
        x = cv.warpPerspective(self.img, mnnM, (480, 360))
        y = cv2.warpPerspective(self.img_, cvM, (480, 360))
        self.assertEqualImg(x, y)
    # miscellaneous
    def test_adaptiveThreshold(self):
        a = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        a_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2GRAY)
        x = cv.adaptiveThreshold(a, 50, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
        y = cv2.adaptiveThreshold(a_, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        self.assertEqual(x.shape[:2], list(y.shape))
    def test_blendLinear(self):
        if version_info.major < 3:
            # py2: cv2 don't have blendLinear
            return
        w1 = np.ones(self.imgf_.shape, dtype=np.float32)
        w2 = w1 * 0.5
        w1_ = expr.const(w1, [1, w1.shape[0], w1.shape[1], 1], expr.NHWC)
        w2_ = w1_ * 0.5
        x = cv.blendLinear(self.imgf, self.imgf, w1_, w2_)
        y = cv2.blendLinear(self.imgf_, self.imgf_, w1, w2)
        self.assertEqualImg(x, y)
    def test_threshold(self):
        x = cv.threshold(self.imgf, 50, 20, cv.THRESH_BINARY)
        y = cv2.threshold(self.imgf_, 50, 20, cv2.THRESH_BINARY)[1]
        self.assertEqualImg(x, y)
    # draw
    def test_Draw(self):
        x = self.img.copy()
        y = self.img_.copy()
        # 1. arrowedLine
        cv.arrowedLine(x, (10, 10), (40, 40), (255, 0, 0))
        cv2.arrowedLine(y, (10, 10), (40, 40), (255, 0, 0))
        # 2. line
        cv.line(x, (20, 30), (50, 60), (0, 0, 255))
        cv2.line(y, (20, 30), (50, 60), (0, 0, 255))
        # 3. circle
        cv.circle(x, (70, 70), 30, (0, 255, 0))
        cv2.circle(y, (70, 70), 30, (0, 255, 0))
        # 4. rectangle
        cv.rectangle(x, (80, 80), (120, 120), (0, 0, 255))
        cv2.rectangle(y, (80, 80), (120, 120), (0, 0, 255))
        # get contours
        y_ = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        y_ = cv2.threshold(y_, 127, 255, cv2.THRESH_BINARY)[1]
        c_, _ = cv2.findContours(y_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = []
        for a in c_:
            ps = []
            for b in a:
                ps.append(int(b[0,0]))
                ps.append(int(b[0,1]))
            c.append(ps)
        # 5. fillPoly
        cv.fillPoly(x, c, [255, 0, 0])
        cv2.fillPoly(y, c_, [255, 0, 0])
        # 6. drawContours
        cv.drawContours(x, c, -1, [0, 0, 255])
        cv2.drawContours(y, c_, -1, [0, 0, 255])
        self.assertEqualImg(x, y)
    # structural
    def test_Structural(self):
        x  = mp.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0,0,0,0],
                        [0,0,1,1,1,1,1,1,1,0,0,0,0],
                        [0,0,1,0,0,1,0,0,0,1,1,0,0],
                        [0,0,1,0,0,1,0,0,1,0,0,0,0],
                        [0,0,1,0,0,1,0,0,1,0,0,0,0],
                        [0,0,1,1,1,1,1,1,1,0,0,0,0],
                        [0,0,0,1,0,0,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],], dtype=mp.uint8)
        x_ = x.read()
        contours, _ = cv.findContours(x, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours_, _ = cv2.findContours(x_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        contour_ = contours_[0]
        self.assertEqualVar(contour, contour_)
        self.assertEqual(cv.contourArea(contour), cv2.contourArea(contour_))
        hull = cv.convexHull(contour)
        hull_ = cv2.convexHull(contour_)
        if version_info.major < 3: hull_ = np.concatenate([hull_[-1::, :], hull_[:-1,:]])
        self.assertEqualVar(hull, hull_)
        rect = cv.minAreaRect(contour)
        rect_ = cv2.minAreaRect(contour_)
        if version_info.major >= 3:
            rect_list = [rect[0][0], rect[0][1], rect[1][0], rect[1][0], rect[2]]
            rect__list = [rect_[0][0], rect_[0][1], rect_[1][0], rect_[1][0], rect_[2]]
            self.assertEqualArray(np.array(rect_list), np.array(rect__list))
        points = cv.boxPoints(rect)
        points_ = cv2.boxPoints(rect_)
        if version_info.major >= 3:
            self.assertEqualVar(points, points_)
        self.assertEqual(tuple(cv.boundingRect(contour)), cv2.boundingRect(contour_))
        ret, labels, statsv, centroids = cv.connectedComponentsWithStats(x)
        ret_, labels_, statsv_, centroids_ = cv2.connectedComponentsWithStats(x_)
        self.assertEqual(ret, ret_)
        labels = expr.squeeze(labels)
        self.assertEqualVar(labels, labels_)
        self.assertEqualVar(statsv, statsv_)
        self.assertEqualVar(centroids, centroids_)
    # histogram
    def test_histogram(self):
        hist = cv.calcHist([self.img], [0], None, [257], [0., 256.])
        hist_ = cv2.calcHist([self.img_], [0], None, [257], [0., 256.])
        self.assertEqualVar(hist, hist_)
    # calib3d
    def test_calib3d(self):
        try:
            a = cv2.SOLVEPNP_SQPNP
        except:
            # python2 opencv don't support SOLVEPNP_SQPNP
            pass
        else:
            model_points = mp.array([0.0, 0.0, 0.0, 0.0, -330.0, -65.0,
                                     -225.0, 170.0, -135.0, 225.0, 170.0, -135.0,
                                     -150.0, -150.0, -125.0, 150.0, -150.0, -125.0]).reshape(6, 3)
            image_points = mp.array([359., 391., 399., 561., 337., 297.,
                                     513., 301., 345., 465., 453., 469.]).reshape(6, 2)
            camera_matrix = mp.array([1200., 0., 600., 0., 1200., 337.5, 0., 0., 1.]).reshape(3, 3)
            dist_coeffs = mp.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
            _, rv, tv = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_SQPNP)
            model_points_ = model_points.read()
            image_points_ = image_points.read()
            camera_matrix_ = camera_matrix.read()
            dist_coeffs_ = dist_coeffs.read()
            _, rv_, tv_ = cv2.solvePnP(model_points_, image_points_, camera_matrix_, dist_coeffs_, flags=cv2.SOLVEPNP_SQPNP)
            self.assertEqualVar(rv, rv_)
            self.assertEqualVar(tv, tv_)
    # core
    def test_vconcat(self):
        x = cv.vconcat([self.img, self.img])
        y = cv2.vconcat([self.img_, self.img_])
        self.assertEqualImg(x, y)
    def test_hconcat(self):
        x = cv.hconcat([self.img, self.img])
        y = cv2.hconcat([self.img_, self.img_])
        self.assertEqualImg(x, y)
    def test_rotate(self):
        x = cv.rotate(self.img, cv.ROTATE_90_CLOCKWISE)
        y = cv2.rotate(self.img_, cv2.ROTATE_90_CLOCKWISE)
        self.assertEqualImg(x, y)
        x = cv.rotate(self.img, cv.ROTATE_180)
        y = cv2.rotate(self.img_, cv2.ROTATE_180)
        self.assertEqualImg(x, y)
        x = cv.rotate(self.img, cv.ROTATE_90_COUNTERCLOCKWISE)
        y = cv2.rotate(self.img_, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.assertEqualImg(x, y)
    def test_solve(self):
        a = np.array([2., 3., 4., 0., 1., 5., 0., 0., 3.]).reshape(3, 3)
        b = np.array([1., 2., 3.]).reshape(3, 1)
        a_ = mp.array([2., 3., 4., 0., 1., 5., 0., 0., 3.]).reshape(3, 3)
        b_ = mp.array([1., 2., 3.]).reshape(3, 1)
        x_, x = cv.solve(a_, b_)
        y_, y = cv2.solve(a, b)
        self.assertEqual(x_, y_)
        self.assertEqualVar(x, y)
    def test_normalize(self):
        x = mp.arange(12).reshape(2, 2, 3).astype(mp.uint8)
        x_ = np.arange(12).reshape(2, 2, 3).astype(np.uint8)
        y = cv.normalize(x, None, -50, 270, cv.NORM_MINMAX)
        y_ = cv2.normalize(x_, None, -50, 270., cv2.NORM_MINMAX)
        self.assertEqualVar(y, y_)
    def test_merge(self):
        # dim = 1
        a = mp.arange(9)
        a_ = np.arange(9)
        channels = [a, a, a]
        channels_ = [a_, a_, a_]
        x = cv.merge(channels)
        y = cv2.merge(channels_)
        self.assertEqualVar(x, y)
        # dim = 2
        a = a.reshape(3, 3)
        a_ = a_.reshape(3, 3)
        channels = [a, a, a]
        channels_ = [a_, a_, a_]
        x = cv.merge(channels)
        y = cv2.merge(channels_)
        self.assertEqualVar(x, y)
        # dim = 3
        channels = [self.img, self.img, self.img]
        channels_ = [self.img_, self.img_, self.img_]
        x = cv.merge(channels)
        y = cv2.merge(channels_)
        self.assertEqualImg(x, y)
        # dim = 5
        a = a.reshape(1, 3, 1, 3, 1)
        a_ = a_.reshape(1, 3, 1, 3, 1)
        channels = [a, a, a]
        channels_ = [a_, a_, a_]
        x = cv.merge(channels)
        y = cv2.merge(channels_)
        self.assertEqualVar(x, y)
    def test_split(self):
        # dim = 1
        a = mp.arange(12.)
        a_ = np.arange(12.)
        channels = cv.split(a)
        channels_ = cv2.split(a_)
        self.assertEqualVars(channels, channels_)
        # dim = 2
        a = a.reshape(4, 3)
        a_ = a_.reshape(4, 3)
        channels = cv.split(a)
        channels_ = cv2.split(a_)
        self.assertEqualVars(channels, channels_)
        # dim = 3
        a = a.reshape(2, 2, 3)
        a_ = a_.reshape(2, 2, 3)
        channels = cv.split(a)
        channels_ = cv2.split(a_)
        self.assertEqualVars(channels, channels_)
        # dim = 4
        a = a.reshape(1, 2, 2, 3)
        a_ = a_.reshape(1, 2, 2, 3)
        channels = cv.split(a)
        channels_ = cv2.split(a_)
        self.assertEqualVars(channels, channels_)
    def test_addWeight(self):
        s = cv.addWeighted(self.img, 0.2, self.img, 0.3, 2)
        s_ = cv2.addWeighted(self.img_, 0.2, self.img_, 0.3, 2)
        self.assertEqualImg(s, s_)
    # numpy
    def test_from_shape_or_value(self):
        x = mp.zeros([2, 2])
        x_ = np.zeros([2, 2])
        self.assertEqualShape(mp.empty([2, 2]).shape, np.empty([2, 2]).shape)
        self.assertEqualShape(mp.empty_like(x).shape, np.empty_like(x_).shape)
        self.assertEqualVar(mp.eye(3, 4, 1), np.eye(3, 4, 1))
        self.assertEqualVar(mp.identity(3), np.identity(3))
        self.assertEqualVar(mp.ones([2, 2]), np.ones([2, 2]))
        self.assertEqualVar(mp.ones_like(x), np.ones_like(x_))
        self.assertEqualVar(mp.zeros([2, 2]), np.zeros([2, 2]))
        self.assertEqualVar(mp.zeros_like(x), np.zeros_like(x_))
        self.assertEqualVar(mp.full([2, 2], 5), np.full([2, 2], 5))
        self.assertEqualVar(mp.full_like(x, 5), np.full_like(x_, 5))
    def test_from_existing_data(self):
        self.assertEqualVar(mp.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertEqualVar(mp.array([1, 2, 3], dtype=mp.float32),
                            np.array([1, 2, 3], dtype=np.float32))
        self.assertEqualVar(mp.array([1, 2, 3], ndmin=5),
                            np.array([1, 2, 3], ndmin=5))
        self.assertEqualVar(mp.asarray([1, 2, 3]), np.asarray([1, 2, 3]))
        self.assertEqualVar(mp.asanyarray([1, 2, 3]), np.asanyarray([1, 2, 3]))
        self.assertEqualVar(mp.ascontiguousarray([1, 2, 3]), np.ascontiguousarray([1, 2, 3]))
        self.assertEqualVar(mp.asmatrix([1, 2, 3]), np.asmatrix([1, 2, 3]))
        self.assertEqualVar(mp.copy(mp.array([1, 2, 3])), np.copy(np.array([1, 2, 3])))
        if version_info.major < 3:
            # py2: don't support `bytes`
            return
        self.assertEqualVar(mp.frombuffer(b'\x01\x02', dtype=mp.uint8), np.frombuffer(b'\x01\x02', dtype=np.uint8))
    def test_numerical_ranges(self):
        self.assertEqualVar(mp.arange(3,7,2), np.arange(3,7,2))
        self.assertEqualVar(mp.arange(5.0), np.arange(5.0))
        self.assertEqualVar(mp.linspace(2.0, 3.0, num=5, endpoint=False), np.linspace(2.0, 3.0, num=5, endpoint=False))
        self.assertEqualVar(mp.logspace(2.0, 3.0, num=4, endpoint=False), np.logspace(2.0, 3.0, num=4, endpoint=False))
        self.assertEqualVar(mp.geomspace(1, 1000, num=4, endpoint=False), np.geomspace(1, 1000, num=4, endpoint=False))
        x = mp.arange(-5, 5., 0.1)
        y = np.arange(-5, 5., 0.1)
        self.assertEqualVars(mp.meshgrid(x, x), np.meshgrid(y, y))
    def test_changing_array_shape(self):
        x = mp.zeros((3, 2))
        x_ = np.zeros((3, 2))
        self.assertEqualShape(mp.shape(x), np.shape(x_))
        self.assertEqualVar(mp.reshape(x, (2, 3)), np.reshape(x_, (2, 3)))
        self.assertEqualVar(mp.ravel(x), np.ravel(x_))
    def test_transpose_like(self):
        x = mp.zeros((3, 4, 5, 6))
        x_ = np.zeros((3, 4, 5, 6))
        self.assertEqualVar(mp.moveaxis(x, 0, -1), np.moveaxis(x_, 0, -1))
        self.assertEqualVar(mp.rollaxis(x, 3, 1), np.rollaxis(x_, 3, 1))
        self.assertEqualVar(mp.swapaxes(x, 0, 2), np.swapaxes(x_, 0, 2))
        self.assertEqualVar(mp.transpose(x), np.transpose(x_))
    def test_changing_number_of_dimensions(self):
        self.assertEqualVars(mp.atleast_1d(1, [2, 3]), np.atleast_1d(1, [2, 3]))
        self.assertEqualVars(mp.atleast_2d(1, [2, 3]), np.atleast_2d(1, [2, 3]))
        self.assertEqualVars(mp.atleast_3d(1, [2, 3]), np.atleast_3d(1, [2, 3]))
        x = mp.array([[1, 2, 3]])
        y = mp.array([[4],[5]])
        x_ = np.array([[1, 2, 3]])
        y_ = np.array([[4],[5]])
        self.assertEqualVar(mp.broadcast_to(x, [3, 3]), np.broadcast_to(x_, [3, 3]))
        self.assertEqualVars(mp.broadcast_arrays(x, y), np.broadcast_arrays(x_, y_))
        self.assertEqualVar(mp.expand_dims(x, axis=0), np.expand_dims(x_, axis=0))
        self.assertEqualVar(mp.squeeze(x, axis=0), np.squeeze(x_, axis=0))
    def test_changing_kind(self):
        self.assertEqualVar(mp.asarray_chkfinite([2, 3]), np.asarray_chkfinite([2, 3]))
        self.assertEqualVar(mp.ascontiguousarray([2, 3]), np.ascontiguousarray([2, 3]))
        if NUMPY_V1: self.assertEqualVar(mp.asfarray([2, 3]), np.asfarray([2, 3])) # removed in numpy 2.0
        try:
            a = np.asscalar
        except:
            # py38 numpy don't support asscalar
            pass
        else:
            self.assertEqual(mp.asscalar(mp.array([24])), np.asscalar(np.array([24])))
    def test_joining(self):
        a = mp.array([1, 2, 3])
        b = mp.array([4, 5, 6])
        a_ = np.array([1, 2, 3])
        b_ = np.array([4, 5, 6])
        self.assertEqualVar(mp.concatenate((a, b)), np.concatenate((a_, b_)))
        self.assertEqualVar(mp.stack((a, b)), np.stack((a_, b_)))
        self.assertEqualVar(mp.vstack((a, b)), np.vstack((a_, b_)))
        self.assertEqualVar(mp.hstack((a, b)), np.hstack((a_, b_)))
        self.assertEqualVar(mp.dstack((a, b)), np.dstack((a_, b_)))
        self.assertEqualVar(mp.column_stack((a, b)), np.column_stack((a_, b_)))
        self.assertEqualVar(mp.row_stack((a, b)), np.row_stack((a_, b_)))
    def test_splitting(self):
        x = mp.arange(9.0)
        x_ = np.arange(9.0)
        self.assertEqualVars(mp.split(x, 3), np.split(x_, 3))
        self.assertEqualVars(mp.split(x, [3, 5, 6, 10]), np.split(x_, [3, 5, 6, 10]))
        self.assertEqualVars(mp.array_split(x, 3), np.array_split(x_, 3))
        x = mp.arange(16.0).reshape([2, 2, 4])
        x_ = np.arange(16.0).reshape([2, 2, 4])
        self.assertEqualVars(mp.dsplit(x, 2), np.dsplit(x_, 2))
        self.assertEqualVars(mp.hsplit(x, 2), np.hsplit(x_, 2))
        self.assertEqualVars(mp.vsplit(x, 2), np.vsplit(x_, 2))
    def test_tiling(self):
        x = mp.array([0, 1, 2])
        x_ = np.array([0, 1, 2])
        self.assertEqualVar(mp.tile(x, 2), np.tile(x_, 2))
        self.assertEqualVar(mp.repeat(x, 2), np.repeat(x_, 2))
    def test_binary_operation(self):
        x = mp.array([1, 2, 3, 4])
        y = mp.array([4, 3, 2, 1])
        x_ = np.array([1, 2, 3, 4])
        y_ = np.array([4, 3, 2, 1])
        self.assertEqualVar(mp.bitwise_and(x, y), np.bitwise_and(x_, y_))
        self.assertEqualVar(mp.bitwise_or(x, y), np.bitwise_or(x_, y_))
        self.assertEqualVar(mp.bitwise_xor(x, y), np.bitwise_xor(x_, y_))
    def test_indexing(self):
        c = mp.array([1, 0, 1, 0])
        c_ = np.array([1, 0, 1, 0])
        self.assertEqualVar(mp.where(c, mp.zeros_like(c, dtype=mp.float32), mp.ones_like(c, dtype=mp.float32)),
                            np.where(c_, np.zeros_like(c_, dtype=np.float32), np.ones_like(c_, dtype=np.float32)))
        self.assertEqualVars(mp.unravel_index([22, 41, 37], (7,6)), np.unravel_index([22, 41, 37], (7,6)))
    def test_linear_algebra(self):
        w = mp.array([1.0, 2.0, 3.0])
        w_ = np.array([1.0, 2.0, 3.0])
        x = mp.arange(18.0).reshape([2,3,3])
        y = mp.arange(12.0).reshape([2,3,2])
        x_ = np.arange(18.0).reshape([2,3,3])
        y_ = np.arange(12.0).reshape([2,3,2])
        z = mp.arange(9).reshape([3,3])
        z_ = np.arange(9).reshape([3,3])
        # 0d * 0d
        self.assertEqual(mp.dot(2, 3), np.dot(2, 3))
        # 1d * 1d
        self.assertEqual(mp.dot(w, w), np.dot(w_, w_))
        # Nd * 1d
        self.assertEqualVar(mp.dot(x, w), np.dot(x_, w_))
        # 2d * 2d
        self.assertEqualVar(mp.dot(z, z), np.dot(z_, z_))
        # Nd * Nd
        self.assertEqualVar(mp.dot(x, y), np.dot(x_, y_))
        self.assertEqualVar(mp.linalg.norm(w), np.linalg.norm(w_))
        a = mp.arange(9.).reshape([3,3])
        a_ = np.arange(9.).reshape([3,3])
        # just compare w Matrix
        self.assertEqualVar(mp.linalg.svd(a)[1], np.linalg.svd(a_)[1])
        a = np.array([2., 3., 4., 0., 1., 5., 0., 0., 3.]).reshape(3, 3)
        b = np.array([1., 2., 3.]).reshape(3, 1)
        a_ = mp.array([2., 3., 4., 0., 1., 5., 0., 0., 3.]).reshape(3, 3)
        b_ = mp.array([1., 2., 3.]).reshape(3, 1)
        x = np.linalg.solve(a, b)
        x_ = mp.linalg.solve(a_, b_)
        self.assertEqualVar(x_, x)

    def test_Logic(self):
        x = mp.array([1,0,2,3,0])
        x_ = np.array([1,0,2,3,0])
        y = mp.array([1,0,2,3,1])
        y_ = np.array([1,0,2,3,1])
        self.assertEqual(mp.all(x), np.all(x_))
        self.assertEqual(mp.any(x), np.any(x_))
        self.assertEqual(mp.array_equal(x, y), np.array_equal(x_, y_))
        self.assertEqual(mp.array_equiv(x, y), np.array_equiv(x_, y_))
        self.assertEqualVar(mp.greater(x, y), np.greater(x_, y_))
        self.assertEqualVar(mp.greater_equal(x, y), np.greater_equal(x_, y_))
        self.assertEqualVar(mp.less(x, y), np.less(x_, y_))
        self.assertEqualVar(mp.less_equal(x, y), np.less_equal(x_, y_))
        self.assertEqualVar(mp.equal(x, y), np.equal(x_, y_))
        self.assertEqualVar(mp.not_equal(x, y), np.not_equal(x_, y_))
    def test_mathematical(self):
        x = mp.array([0.1, 0.3, 0.5, 0.7])
        y = mp.array([1.2, 1.4, 1.6, 1.8])
        x_ = np.array([0.1, 0.3, 0.5, 0.7])
        y_ = np.array([1.2, 1.4, 1.6, 1.8])
        self.assertEqualVar(mp.sin(x), np.sin(x_))
        self.assertEqualVar(mp.cos(x), np.cos(x_))
        self.assertEqualVar(mp.tan(x), np.tan(x_))
        self.assertEqualVar(mp.arcsin(x), np.arcsin(x_))
        self.assertEqualVar(mp.arccos(x), np.arccos(x_))
        self.assertEqualVar(mp.arctan(x), np.arctan(x_))
        self.assertEqualVar(mp.hypot(x, y), np.hypot(x_, y_))
        self.assertEqualVar(mp.arctan2(x, y), np.arctan2(x_, y_))
        self.assertEqualVar(mp.sinh(x), np.sinh(x_))
        self.assertEqualVar(mp.cosh(x), np.cosh(x_))
        self.assertEqualVar(mp.tanh(x), np.tanh(x_))
        self.assertEqualVar(mp.arcsinh(x), np.arcsinh(x_))
        self.assertEqualVar(mp.arccosh(y), np.arccosh(y_))
        self.assertEqualVar(mp.arctanh(x), np.arctanh(x_))
        self.assertEqualVar(mp.arctanh(x), np.arctanh(x_))
        self.assertEqualVar(mp.around(y), np.around(y_))
        if NUMPY_V1: self.assertEqualVar(mp.round_(y), np.round_(y_))
        self.assertEqualVar(mp.rint(y), np.rint(y_))
        # self.assertEqualVar(mp.fix(y), np.fix(y_))
        self.assertEqualVar(mp.floor(y), np.floor(y_))
        self.assertEqualVar(mp.ceil(y), np.ceil(y_))
        # self.assertEqualVar(mp.trunc(y), np.trunc(y_))
        self.assertAlmostEqual(mp.prod(x), np.prod(x_), delta=1e-3)
        self.assertAlmostEqual(mp.sum(x), np.sum(x_), delta=1e-3)
        self.assertAlmostEqual(mp.nanprod(x), np.nanprod(x_), delta=1e-3)
        self.assertAlmostEqual(mp.nansum(x), np.nansum(x_), delta=1e-3)
        self.assertEqualVar(mp.sqrt(x), np.sqrt(x_))
        self.assertEqualVar(mp.exp(x), np.exp(x_))
        self.assertEqualVar(mp.exp2(x), np.exp2(x_))
        self.assertEqualVar(mp.expm1(x), np.expm1(x_))
        self.assertEqualVar(mp.log(x), np.log(x_))
        self.assertEqualVar(mp.log2(x), np.log2(x_))
        self.assertEqualVar(mp.log10(x), np.log10(x_))
        self.assertEqualVar(mp.log1p(x), np.log1p(x_))
        self.assertEqualVar(mp.logaddexp(x, y), np.logaddexp(x_, y_))
        self.assertEqualVar(mp.logaddexp2(x, y), np.logaddexp2(x_, y_))
        self.assertEqualVar(mp.sinc(x), np.sinc(x_))
        self.assertEqualVar(mp.sign(x), np.sign(x_))
        self.assertEqualVar(mp.signbit(x), np.signbit(x_))
        self.assertEqualVar(mp.copysign(x, y), np.copysign(x_, y_))
        self.assertEqualVar(mp.ldexp(x, 2.0), np.ldexp(x_, 2))
        self.assertEqualVar(mp.reciprocal(x), np.reciprocal(x_))
        self.assertEqualVar(mp.positive(x), np.positive(x_))
        self.assertEqualVar(mp.negative(x), np.negative(x_))
        self.assertEqualVar(mp.add(x, y), np.add(x_, y_))
        self.assertEqualVar(mp.multiply(x, y), np.multiply(x_, y_))
        self.assertEqualVar(mp.power(x, y), np.power(x_, y_))
        self.assertEqualVar(mp.subtract(x, y), np.subtract(x_, y_))
        self.assertEqualVar(mp.true_divide(x, y), np.true_divide(x_, y_))
        self.assertEqualVar(mp.floor_divide(x, y), np.floor_divide(x_, y_))
        self.assertEqualVar(mp.float_power(x, y), np.float_power(x_, y_))
        self.assertEqualVar(mp.mod(x, y), np.mod(x_, y_))
        self.assertEqualVar(mp.fmod(x, y), np.fmod(x_, y_))
        self.assertEqualVar(mp.remainder(x, y), np.remainder(x_, y_))
        self.assertEqualVars(mp.divmod(x, y), np.divmod(x_, y_))
        self.assertEqualVars(mp.modf(x), np.modf(x_))
        self.assertEqualVar(mp.clip(x, 0.3, 0.6), np.clip(x_, 0.3, 0.6))
        self.assertEqualVar(mp.cbrt(x), np.cbrt(x_))
        self.assertEqualVar(mp.square(x), np.square(x_))
        self.assertEqualVar(mp.abs(x), np.abs(x_))
        self.assertEqualVar(mp.absolute(x), np.absolute(x_))
        self.assertEqualVar(mp.fabs(x), np.fabs(x_))
        self.assertEqualVar(mp.maximum(x, y), np.maximum(x_, y_))
        self.assertEqualVar(mp.minimum(x, y), np.minimum(x_, y_))
        self.assertEqualVar(mp.fmax(x, y), np.fmax(x_, y_))
        self.assertEqualVar(mp.fmin(x, y), np.fmin(x_, y_))
        self.assertEqualVar(mp.cumprod(x), np.cumprod(x_))
        self.assertEqualVar(mp.cumsum(x), np.cumsum(x_))
    def test_matrix(self):
        import numpy.matlib
        self.assertEqualVar(mp.repmat([1, 2, 3], 2, 2), np.matlib.repmat([1, 2, 3], 2, 2))
    def test_padding(self):
        x = mp.array([[1,2],[3,4]])
        x_ = np.array([[1,2],[3,4]])
        self.assertEqualVar(mp.pad(x, 2, 'constant'), np.pad(x_, 2, 'constant'))
        #self.assertEqualVar(mp.pad(x, (2,3), 'reflect'), np.pad(x_, (2,3), 'reflect'))
        #self.assertEqualVar(mp.pad(x, ((2,3),(1,4)), 'symmetric'), np.pad(x_, ((2,3),(1,4)), 'symmetric'))
    def test_rand(self):
        self.assertEqualShape(mp.random.random([3,3]).shape, np.random.random([3,3]).shape)
        self.assertEqualShape(mp.random.randn(2,3).shape, np.random.randn(2,3).shape)
        self.assertEqualShape(mp.random.rand(3,2).shape, np.random.rand(3,2).shape)
        self.assertEqualShape(mp.random.randint(0, 2, [2,3]).shape, np.random.randint(0, 2, [2,3]).shape)
    def test_sorting(self):
        x = mp.array([[1,0,3], [0,6,5]])
        x_ = np.array([[1,0,3], [0,6,5]])
        self.assertEqualVar(mp.sort(x), np.sort(x_))
        self.assertEqualVar(mp.argsort(x), np.argsort(x_))
    def test_searching_counting(self):
        x = mp.array([[1,0,3], [0,6,5]])
        x_ = np.array([[1,0,3], [0,6,5]])
        self.assertEqual(mp.argmax(x), np.argmax(x_))
        self.assertEqual(mp.argmin(x), np.argmin(x_))
        self.assertEqualVar(mp.argwhere(x), np.argwhere(x_))
        self.assertEqualVars(mp.nonzero(x), np.nonzero(x_))
        self.assertEqualVar(mp.flatnonzero(x), np.flatnonzero(x_))
        self.assertEqualVar(mp.count_nonzero(x, 0), np.count_nonzero(x_, 0))
    def test_statistics(self):
        x = mp.array([[1,2,3], [4,6,5]])
        x_ = np.array([[1,2,3], [4,6,5]])
        self.assertEqual(mp.amin(x), np.amin(x_))
        self.assertEqualVar(mp.amin(x, 0), np.amin(x_, 0))
        self.assertEqual(mp.amax(x), np.amax(x_))
        self.assertEqualVar(mp.amax(x, 0), np.amax(x_, 0))
        if NUMPY_V1:
            self.assertEqual(mp.ptp(x), np.ptp(x_))
            self.assertEqualVar(mp.ptp(x, 1), np.ptp(x_, 1))
        self.assertAlmostEqual(mp.mean(x), np.mean(x_), delta=1e-3)
        self.assertEqualVar(mp.mean(x, 0), np.mean(x_,0))
        self.assertAlmostEqual(mp.var(x), np.var(x_), delta=1e-3)
        self.assertEqualVar(mp.var(x, 1), np.var(x_,1))
        self.assertAlmostEqual(mp.std(x), np.std(x_), delta=1e-3)
        self.assertEqualVar(mp.std(x, 0), np.std(x_,0))
    def test_histogram(self):
        x = mp.array([1., 2., 2., 3., 3., 3., 4., 4., 5.])
        x_ = x.read()
        self.assertEqualVars(mp.histogram(x, 7, (2, 4)), np.histogram(x_, 7, (2, 4)))
    def test_ndarray(self):
        x = mp.array([[1,2],[3,4]])
        x_ = np.array([[1,2],[3,4]])
        self.assertEqual(x.all(), x_.all())
        self.assertEqual(x.any(), x_.any())
        self.assertEqual(x.argmax(), x_.argmax())
        self.assertEqualVar(x.argmax(0), x_.argmax(0))
        self.assertEqualVar(x.astype(mp.float32), x_.astype(np.float32))
        self.assertEqualVar(x.clip(0,3), x_.clip(0,3))
        self.assertEqualVar(x.dot(x), x_.dot(x_))
        y = x.copy()
        y_ = x_.copy()
        self.assertEqualVar(y, y_)
        y.fill(3)
        y_.fill(3)
        self.assertEqualVar(y, y_)
        self.assertEqualVar(x.flatten(), x_.flatten())
        self.assertEqual(x.max(), x_.max())
        self.assertEqualVar(x.max(0), x_.max(0))
        self.assertEqual(x.mean(), x_.mean())
        self.assertEqualVar(x.mean(0), x_.mean(0))
        self.assertEqual(x.min(), x_.min())
        self.assertEqualVar(x.min(0), x_.min(0))
        self.assertEqualVars(x.nonzero(), x_.nonzero())
        self.assertEqual(x.prod(), x_.prod())
        self.assertEqualVar(x.prod(0), x_.prod(0))
        if NUMPY_V1:
            self.assertEqual(x.ptp(), x_.ptp())
            self.assertEqualVar(x.ptp(0), x_.ptp(0))
        self.assertEqualVar(x.ravel(), x_.ravel())
        self.assertEqualVar(x.repeat(2), x_.repeat(2))
        self.assertEqualVar(x.reshape([4,1]), x_.reshape([4,1]))
        self.assertEqualVar(x.squeeze(), x_.squeeze())
        self.assertAlmostEqual(x.std(), x_.std())
        self.assertEqualVar(x.std(0), x_.std(0))
        self.assertEqual(x.sum(), x_.sum())
        self.assertEqualVar(x.sum(0), x_.sum(0))
        self.assertEqualVar(x.swapaxes(0, 1), x_.swapaxes(0, 1))
        self.assertEqualVar(x.transpose(), x_.transpose())
        self.assertAlmostEqual(x.var(), x_.var())
        self.assertEqualVar(x.var(0), x_.var(0))
        self.assertEqual(len(x), len(x_))
        self.assertEqual(x[0,1], x_[0,1])
        self.assertEqualVar(x[0], x_[0])
        self.assertEqualVar(x[:], x_[:])
        self.assertEqualVar(x[:1], x_[:1])
        self.assertEqualVar(x[::-1], x_[::-1])
        self.assertEqualVar(x[x > 2], x_[x_ > 2])
        self.assertEqualVar(x[mp.array([1])], x_[np.array([1])])
if __name__ == '__main__':
    unittest.main()
