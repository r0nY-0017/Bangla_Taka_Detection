import os
import cv2
import numpy as np
import pickle

# ORB ডিটেক্টর
orb = cv2.ORB_create(nfeatures=2000)
FIXED_FEATURE_SIZE = 27232  # ফিচার সাইজ ফিক্সড

# ডেটা লোড এবং ফিচার এক্সট্রাকশন
data = []
labels = []
class_names = ['1000', '500', '200', '100', '50', '20', '10']

for label in class_names:
    folder = f'D:\Taka_Detection\Taka_Images\Training\{label}'
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (660, 380))
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            features = descriptors.flatten()
            # ফিচার সাইজ ফিক্সড করা
            if len(features) < FIXED_FEATURE_SIZE:
                features = np.pad(features, (0, FIXED_FEATURE_SIZE - len(features)), mode='constant')
            else:
                features = features[:FIXED_FEATURE_SIZE]
            data.append(features)
            labels.append(class_names.index(label))
        else:
            print(f"No descriptors found for {img_path}")

# Pickle ফাইলে সেভ
data_dict = {'data': data, 'labels': labels}
with open('taka.pickle', 'wb') as f:
    pickle.dump(data_dict, f)

print("Feature extraction completed and saved to taka.pickle")