import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.api.keras.preprocessing import image

class inferenceTF():
    def __init__(self, imgPath, frozen_model_filename, imgSizeH, imgSizeW):
        self.img = imgPath
        self.model = frozen_model_filename
        self.height = imgSizeH
        self.width = imgSizeW
        self.graph = self.load_graph(self.model)
        self.input_data = None

    def ZeroCenter(self, path, sizeH, sizeW, BGRTranspose=False):
        img = image.load_img(path, target_size=(self.height, self.width))
        x = image.img_to_array(img)

        # Reference: 1) Keras image preprocess: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
        #            2) tensorflow github issue: https://github.com/tensorflow/models/issues/517
        # R-G-B for Imagenet === [123.68, 116.78, 103.94]

        # x[..., 0] -= 123.68
        # x[..., 1] -= 116.779
        # x[..., 2] -= 103.939
        # x[..., 0] = 50 / 255
        # x[..., 1] = 50 / 255
        # x[..., 2] = 50 / 255
        # x[..., 0] /= 255.0
        # x[..., 1] /= 255.0
        # x[..., 2] /= 255.0

        if BGRTranspose == True:
            x = x[..., ::-1]

        return x

    def load_graph(self, frozen_graph_filename):
        # parse the graph_def file
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # load the graph_def in the default graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def save2file(self, file_name, tensorArr, C, H, W):
        """
        c
        h
        w
        data
        .
        .
        .
        """
        f = open(file_name, 'w')
        # f.write(str(C) + '\n')
        # f.write(str(H) + '\n')
        # f.write(str(W) + '\n')
        # for c in range(C):
        #     for h in range(H):
        #         for w in range(W):
        #             if len(tensorArr.shape) == 4:
        #                 f.write(str(tensorArr[0][h][w][c]) + "\n")
        #             else:
        #                 f.write(str(tensorArr[0][c]) + "\n")
        # for c in range(C):
        #     for h in range(H):
        #         for w in range(W):
        #             if len(tensorArr.shape) == 4:
        #                 f.write(str(tensorArr[0][h][w][c]) + "\t")
        #             else:
        #                 f.write(str(tensorArr[0][c]))
        #         f.write("\n")
        # f.close()
        ### NHWC
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    if len(tensorArr.shape) == 4:
                        f.write(str(tensorArr[0][h][w][c]) + "\t")
                    else:
                        f.write(str(tensorArr[0][c]))
                f.write("\n")
        f.close()

    def saveTensorByName(self, input_node_name, layer_name, file_name):
        img = self.ZeroCenter(self.img, self.height, self.width)
        img = np.expand_dims(img, axis=0)

        init = tf.global_variables_initializer()

        data_input = self.graph.get_tensor_by_name(input_node_name)
        for_test = self.graph.get_tensor_by_name(layer_name)

        with tf.Session(graph = self.graph) as sess:
            res = sess.run(for_test, feed_dict={data_input: img})
            print(">" * 50)
            print(res.shape)
            print(res)
            if len(res.shape) == 4:
                n, h, w, c = res.shape
            else:
                n, c = res.shape
                h, w = (1, 1)
            self.save2file(file_name, res, c, h, w)

        return 0

    def generateMNNInput(self, input_node_name, file_name):
        img = self.ZeroCenter(self.img, self.height, self.width)
        img = np.expand_dims(img, axis=0)

        init = tf.global_variables_initializer()

        data_input = self.graph.get_tensor_by_name(input_node_name)

        with tf.Session(graph = self.graph) as sess:

            self.input_data = sess.run(data_input, feed_dict={data_input: img})

        f = open(file_name, "w")
        for c in range(3):
            for h in range(self.height):
                for w in range(self.width):
                    f.write(str(self.input_data[0][h][w][c]) + '\n')

        # for h in range(self.height):
        #     for w in range(self.width):
        #         for c in range(3):
        #             f.write(str(self.input_data[0][h][w][c]) + '\n')

        f.close()


if __name__ == '__main__':
    img = './test.jpg'
    frozen_model_filename = 'path/to/model.pb'
    imgSizeH = 256
    imgSizeW = 256
    inputTensorName = "inputs:0"
    outputTensorName = "output:0"
    test = inferenceTF(img, frozen_model_filename, imgSizeH, imgSizeW)
    test.generateMNNInput(inputTensorName, "input_0.txt")
    test.saveTensorByName(inputTensorName, outputTensorName, "TF_Result.txt")
