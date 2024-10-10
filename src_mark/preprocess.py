from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import os
import joblib
import new_preprocess
from skimage import io

data_dir = 'dataset-jerry'
data = []
labels = []
categories = os.listdir(data_dir)
print("Loading data...")

for category in categories:

    class_ID = int(category)  # Convert folder name to integer
    folder_path = os.path.join(data_dir, category)

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = io.imread(img_path)
        img = new_preprocess.preprocess_image(img)
        img_np = np.array(img)
        img_flatten = img_np.flatten()
        data.append(img_flatten)
        labels.append(class_ID)

data = np.array(data)
labels = np.array(labels)

data = data / 255.0
print("Data loaded successfully...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

joblib.dump((X_train, y_train), 'src_mark/training_data.joblib')
joblib.dump((X_test, y_test), 'src_mark/testing_data.joblib')