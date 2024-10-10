import skimage
import cv2
from skimage import io, color, transform
import numpy as np
import joblib
import matplotlib.pyplot as plt
import new_preprocess

x = 112

def bgr8_to_rgb8(snapshot):
    temp_blue = snapshot[:,:,0]
    temp_red = snapshot[:,:,2]
    snapshot[:,:,0] = temp_red
    snapshot[:,:,2] = temp_blue
    return snapshot
    
def flatten_and_normalize(X):
    X_flat = X.reshape(1, -1)
    return X_flat.astype('float32') / 255.0
    
# def load_and_preprocess(img_path):
#     img = Image.open(img_path)
def load_and_preprocess(snapshot):
    img = preprocess_image(snapshot)
    img_flat = flatten_and_normalize(np.array(img))
    return img_flat

def preprocess_image(img):
    img_cropped = img[50:(224-30), 40:(224-60)]
    # img_cropped = img
    # convert to lab colorspace
    img_LAB = skimage.color.rgb2lab(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

    dark_mask = (img_LAB[:, :, 0] < 50) & (img_LAB[:, :, 1] < 22) # road
    light_mask = (img_LAB[:, :, 0] > 50) & (img_LAB[:, :, 1] > -40) & (img_LAB[:, :, 1] < 33)
    img_LAB[dark_mask | light_mask] = [100, 50, -5]
    img_RGB = skimage.color.lab2rgb(img_LAB)
    img_resized = skimage.transform.resize(img_RGB, (x, x))
    img_gray = np.clip(img_resized[:, :, 0] - img_resized[:, :, 1], 0, 255)
    kernel = np.ones((2, 2), np.uint8)
    img_filtered = cv2.morphologyEx(skimage.img_as_ubyte(img_gray), cv2.MORPH_CLOSE, kernel)

    # Slower but better noise removal
    # img_filtered = cv2.fastNlMeansDenoising(img_filtered, None, 17, 9, 21)
    return img_filtered

if __name__ == '__main__':
    # img = io.imread('dataset-jerry/5/stop (700).jpg')
    img = io.imread('dataset-jerry/2/road (400).jpg')
    img = io.imread('dataset-jerry/3/sheep (1000).jpg')
    # img = io.imread('dataset-jerry/4/speed (10).jpg')
    # img = io.imread('dataset-jerry/0/green (1).jpg')
    img = io.imread('dataset-jerry/1/red (1).jpg')


    io.imshow(img)
    plt.show()

    # io.imshow(img_processed)
    # plt.show()
    img = new_preprocess.preprocess_image(img)
    io.imshow(img)
    plt.show()
    img_np = np.array(img)
    img_flatten = img_np.flatten()
    # img_processed = img_flatten
    img_processed = img_flatten / 255.0
    img_processed = img_processed.reshape(1, -1)


    model = joblib.load('src_mark\model-anthony.joblib')
    prediction = model.predict(img_processed)
    print("-----------------------------")
    print(prediction)

