import numpy as np
import os
from PIL import Image
import MNN
F = MNN.expr


# adapted from pycaffe
def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def center_crop(image_data, crop_factor):
    height, width, channels = image_data.shape

    h_size = int(height * crop_factor)
    h_start = int((height - h_size) / 2)
    h_end = h_start + h_size

    w_size = int(width * crop_factor)
    w_start = int((width - w_size) / 2)
    w_end = w_start + w_size

    cropped_image = image_data[h_start:h_end, w_start:w_end, :]

    return cropped_image


def resize_image(image, shape):
    im = Image.fromarray(image)
    im = im.resize(shape)
    resized_image = np.array(im)

    return resized_image


class CalibrationDataset(MNN.data.Dataset):
    '''
    This is demo for Imagenet calibration dataset. like pytorch, you need to overload __getiterm__ and __len__ methods
    __getiterm__ should return a sample in F.const, and you should not use batch dimension here
    __len__ should return the number of total samples in the calibration dataset
    '''
    def __init__(self, image_folder):
        super(CalibrationDataset, self).__init__()
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)[0:1000]

    def __getitem__(self, index):
        image_name = os.path.join(self.image_folder, self.image_list[index].split(' ')[0])


        # preprocess your data here, the following code are for tensorflow mobilenets
        image_data = load_image(image_name)
        image_data = center_crop(image_data, 0.875)
        image_data = resize_image(image_data, (224, 224))
        image_data = (image_data - 127.5) / 127.5


        # after preprocessing the data, convert it to MNN data structure
        dv = F.const(image_data.flatten().tolist(), [224, 224, 3], F.data_format.NHWC, F.dtype.float)

        '''
        first list for inputs, and may have many inputs, so it's a list
        if your model have more than one inputs, add the preprocessed MNN const data to the input list

        second list for targets, also, there may be more than one targets
        for calibration dataset, we don't need labels, so leave it blank

        Note that, the input order in the first list should be the same in your 'config.yaml' file.
        '''
        
        return [dv], []

    def __len__(self):
        # size of the dataset
        return len(self.image_list)


'''
initialize a CalibrationDataset object, its name should be exactly 'calibration_dataset'
'''
calibration_dataset = CalibrationDataset(image_folder='/data/imagenet_train_images')
