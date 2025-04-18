import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ডেটা লোড এবং প্রি-প্রসেসিং
data = []
labels = []
class_names = ['1000', '500', '200', '100', '50', '20', '10']
IMG_SIZE = (224, 224)

for label in class_names:
    folder = f'D:\Taka_Detection\Taka_Images\Training\{label}'
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # নরমালাইজেশন
        data.append(img)
        labels.append(class_names.index(label))

data = np.array(data)
labels = np.array(labels)

# ট্রেনিং এবং টেস্টিং সেটে ভাগ
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN মডেল তৈরি
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# মডেল কম্পাইল
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ডেটা অগমেন্টেশন
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# ট্রেনিং (History সংরক্ষণ করা হচ্ছে গ্রাফের জন্য)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=2, 
                    validation_data=(X_test, y_test))

# মডেল সেভ
model.save('banknote_cnn_model.h5')

# টেস্টিং নির্ভুলতা
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Accuracy এবং Loss গ্রাফ
def plot_training_history(history):
    # Accuracy Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(model, X_test, y_test, class_names):
    # টেস্ট ডেটার প্রেডিকশন
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion Matrix তৈরি
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # প্লট
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Classification Report (অতিরিক্ত তথ্য)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

# গ্রাফ প্লট করা
plot_training_history(history)
plot_confusion_matrix(model, X_test, y_test, class_names)