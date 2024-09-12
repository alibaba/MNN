import os
def convert(onnx_path, mnn_path, extra):
    print('Onnx path: ', onnx_path)
    print('MNN path: ', mnn_path)
    print('Extra: ', extra)
    convert_path = '../../../build/MNNConvert'
    if not os.path.exists(convert_path):
        print(convert_path + " not exist, use pymnn instead")
        convert_path = 'mnnconvert'
    models = ['text_encoder', 'unet', 'vae_decoder']
    for model in models:
        cmd = convert_path + ' -f ONNX --modelFile ' + os.path.join(onnx_path, model, 'model.onnx') + ' --MNNModel ' + os.path.join(mnn_path, model + '.mnn') + ' --saveExternalData=1 ' + extra
        print(cmd)
        print(os.popen(cmd).read())

if __name__ == '__main__':
    import sys
    extra = ""
    if len(sys.argv) > 3:
        extra = sys.argv[3]
    convert(sys.argv[1], sys.argv[2], extra)
