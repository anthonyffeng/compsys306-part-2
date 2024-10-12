import numpy as np
import cv2
import skimage
from skimage import io, color, transform
import matplotlib.pyplot as plt

x = 112

def preprocess_image(img):
    img_cropped = img[70:(224-30), 40:(224-40)]
    img_LAB = skimage.color.rgb2lab(img_cropped)
    # this is the black line mask
    dark_mask =  (img_LAB[:, :, 0] < 55) & (img_LAB[:, :, 1] < 10) & (img_LAB[:, :, 1]  > -40) | (img_LAB[:, :, 2] < 0)

    # this is the white space mask
    light_mask = (img_LAB[:, :, 0] > 55) & (img_LAB[:, :, 1] > -40) & (img_LAB[:, :, 1] < 40)
    
    # convert background to blue
    img_LAB[dark_mask | light_mask] = [32.302586667249486, 79.19666178930935, -107.86368104495168]
    img_RGB = skimage.color.lab2rgb(img_LAB)
    kernel = np.ones((2, 2), np.uint8)
    img_filtered = cv2.morphologyEx(skimage.img_as_ubyte(img_RGB), cv2.MORPH_CLOSE, kernel)
    return img_filtered



if __name__ == "__main__":
    img = io.imread('src_mark/sheep.jpg')
    img = io.imread('src_mark/sheep_left.jpg')
    img = io.imread('src_mark/sheep_right.jpg')
    img = io.imread('src_mark/speed.jpg')
    img = io.imread('src_mark/speed(1).jpg')
    img = io.imread('src_mark/stop.jpg')
    img = io.imread('src_mark/stop_right (2).jpg')
    img = io.imread('src_mark/red.jpg')
    img = io.imread('src_mark/green.jpg')


    img = io.imread('dataset-jerry/3/sheep (1000).jpg')
    
    io.imshow(img)
    plt.show()

    # img_cropped = img[70:(224-30), 40:(224-40)]
    # io.imshow(img_cropped)
    # plt.show()
    # img_LAB = skimage.color.rgb2lab(img_cropped)
    # # this is the black line mask
    # dark_mask =  (img_LAB[:, :, 0] < 55) & (img_LAB[:, :, 1] < 10) & (img_LAB[:, :, 1]  > -40) | (img_LAB[:, :, 2] < 0)

    # # this is the white space mask
    # light_mask = (img_LAB[:, :, 0] > 55) & (img_LAB[:, :, 1] > -40) & (img_LAB[:, :, 1] < 40)
    
    # # convert background to blue
    # img_LAB[dark_mask | light_mask] = [32.302586667249486, 79.19666178930935, -107.86368104495168]
    # img_RGB = skimage.color.lab2rgb(img_LAB)
    # kernel = np.ones((2, 2), np.uint8)
    # img_filtered = cv2.morphologyEx(skimage.img_as_ubyte(img_RGB), cv2.MORPH_CLOSE, kernel)

    img_RGB = preprocess_image(img)
    io.imshow(img_RGB)
    plt.show()