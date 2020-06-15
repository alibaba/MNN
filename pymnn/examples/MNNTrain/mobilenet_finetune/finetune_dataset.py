import numpy as np
import MNN
from PIL import Image
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


def resize_image(image, shape):
    im = Image.fromarray(image)
    im = im.resize(shape)
    resized_image = np.array(im)

    return resized_image


class FinetuneDataset(MNN.data.Dataset):
    def __init__(self, image_folder, image_txt):
        super(FinetuneDataset, self).__init__()
        self.image_folder = image_folder
        self.image_txt = image_txt
        f = open(image_txt)
        self.image_list = f.readlines()

    def __getitem__(self, index):
        image_name = self.image_list[index].split(" ")[0]
        image_label = int(self.image_list[index].split(" ")[1])
        image_file = self.image_folder + image_name

        image_data = load_image(image_file)
        image_data = resize_image(image_data, (224, 224))

        image_data = (image_data - 127.5) / 127.5

        dv = F.const(image_data.flatten().tolist(), [224, 224, 3], F.data_format.NHWC)
        dl = F.const([image_label], [], F.data_format.NHWC, F.dtype.int)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        return len(self.image_list)
