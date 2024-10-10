import numpy as np
import cv2
import skimage
from skimage import io, color, transform
import matplotlib.pyplot as plt

x = 112

def preprocess_image(img):
    img_cropped = img[50:(224-30), 40:(224-60)]
    # convert to lab colorspace
    img_LAB = skimage.color.rgb2lab(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    dark_mask = (img_LAB[:, :, 0] < 50) & (img_LAB[:, :, 1] < 10)
    light_mask = (img_LAB[:, :, 0] > 50) & (img_LAB[:, :, 1] > -40) & (img_LAB[:, :, 1] < 33)
    img_LAB[dark_mask | light_mask] = [100, 50, -5]
    img_RGB = skimage.color.lab2rgb(img_LAB)
    # img_resized = skimage.transform.resize(img_RGB, (x, x))
    img_resized = img_RGB
    img_gray = np.clip(img_resized[:, :, 0] - img_resized[:, :, 1], 0, 255)
    kernel = np.ones((2, 2), np.uint8)
    img_filtered = cv2.morphologyEx(skimage.img_as_ubyte(img_gray), cv2.MORPH_CLOSE, kernel)

    # Slower but better noise removal
    # img_filtered = cv2.fastNlMeansDenoising(img_filtered, None, 17, 9, 21)
    return img_filtered


if __name__ == "__main__":
    img = io.imread('dataset-jerry/2/road (400).jpg')
    img = io.imread('dataset-jerry/3/sheep (100).jpg')
    # img = io.imread('dataset-jerry/5/stop (10).jpg')
    # img = io.imread('dataset-jerry/4/speed (1000).jpg')
    # img = io.imread('dataset-jerry/0/green (1).jpg')
    # img = io.imread('dataset-jerry/1/red (1).jpg')
    # img = io.imread('image.png')
    # io.imshow(img)
    # plt.show()

    img_cropped = img[50:(224-30), 40:(224-60)]
    io.imshow(img_cropped)
    plt.show()

    img_processed = preprocess_image(img)
    print(img_processed.shape)
    print(img_processed[10,10])
    io.imshow(img_processed)
    plt.show()