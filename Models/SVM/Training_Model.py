import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ডেটা লোড
with open('taka_svm.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    data = data_dict['data']
    labels = data_dict['labels']

# ডেটার আকৃতি চেক
data = np.array(data)
labels = np.array(labels)
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# ট্রেনিং এবং টেস্টিং সেটে ভাগ
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# SVM মডেল
model = SVC(kernel='rbf', probability=True, random_state=42)  # RBF কার্নেল ব্যবহার
model.fit(X_train, y_train)

# ট্রেনিং এবং টেস্টিং নির্ভুলতা
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# মডেল সেভ
with open('model_svm.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Accuracy গ্রাফ
def plot_accuracy_graph(train_accuracy, test_accuracy):
    plt.figure(figsize=(6, 4))
    plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
    plt.title('Model Accuracy (SVM)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate([train_accuracy, test_accuracy]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_test, test_pred, class_names):
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (SVM)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=class_names))

# গ্রাফ প্লট
class_names = ['1000', '500', '200', '100', '50', '20', '10']
plot_accuracy_graph(train_accuracy, test_accuracy)
plot_confusion_matrix(y_test, test_pred, class_names)