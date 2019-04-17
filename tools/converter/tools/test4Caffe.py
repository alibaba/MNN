from __future__ import print_function
import argparse
import numpy as np
import sys
sys.path.append('path/to/caffe/python')
import caffe


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


def eval():
    nh, nw = 224, 224
    img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)

    caffe.set_mode_cpu()
    net = caffe.Net(args.proto, args.model, caffe.TEST)

    im = caffe.io.load_image(args.image)
    h, w, _ = im.shape
    if h < w:
        off = (int)((w - h) / 2)
        im = im[:, off:off + h]
    else:
        off = (h - w) / 2
        im = im[off:off + h, :]
    im = caffe.io.resize_image(im, [nh, nw])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_mean('data', img_mean)
    transformer.set_input_scale('data', 0.017)

    net.blobs['data'].reshape(1, 3, nh, nw)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    prob = out['prob']
    prob = np.squeeze(prob)
    # weight = net.params['conv1'][0].data
    # print(weight.shape)
    # print(weight[0, :, :, :])
    # print('=' * 100)

    testforMNN = net.blobs['conv2_1/dw'].data
    print(testforMNN.shape)
    print(testforMNN[0,1,:,:])


    idx = np.argsort(-prob)
    imgData = net.blobs['data'].data
    f = open("input.txt", "w")
    _, c, h, w = imgData.shape
    for ci in range(c):
        for hi in range(h):
            for wi in range(w):
                f.write(str(imgData[0][ci][hi][wi]) + '\n')
    f.close()
    label_names = np.loadtxt('synset.txt', str, delimiter='\t')
    for i in range(5):
        label = idx[i]
        print('%.2f - %s' % (prob[label], label_names[label]))
    return


if __name__ == '__main__':
    eval()
