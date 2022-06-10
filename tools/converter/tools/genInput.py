import numpy as np
from PIL import Image
if __name__ == '__main__':
    img = './test.jpg'
    imgSizeH = 256
    imgSizeW = 256
    with Image.open(img) as im:
        im = im.resize((imgSizeW, imgSizeH))
        np.savetxt("input_0.txt", im.getdata())
