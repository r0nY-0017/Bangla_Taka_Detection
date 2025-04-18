import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk
import pytesseract
import logging
import re
import os

# Tesseract পথ
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

# ORB Detector
orb = cv2.ORB_create(nfeatures=2000)

# Label dictionary
labels_dict = {
    0: '1000 Taka',
    1: '500 Taka',
    2: '200 Taka',
    3: '100 Taka',
    4: '50 Taka',
    5: '20 Taka',
    6: '10 Taka',
}

# Threshold settings
MIN_KEYPOINTS = 20
CONFIDENCE_THRESHOLD = 0.4

# OCR ফাংশন
def extract_number_from_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "temp_image.png")
        cv2.imwrite(temp_file, binary)
        
        text = pytesseract.image_to_string(temp_file, config='--psm 6 digits')
        print(f"Raw OCR output: {text}")
        try:
            os.remove(temp_file)
        except PermissionError:
            logging.warning("Could not delete temp file.")
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[0]
        return "Number not detected"
    except Exception as e:
        logging.error(f"OCR Error: {e}")
        return "OCR Error"

# GUI Class
class BanknoteDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Banknote Detector (ORB + Random Forest)")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Select an image to detect banknote", font=("Arial", 14))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Image", command=self.select_image, font=("Arial", 12))
        self.select_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=10)

        self.number_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.number_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Banknote Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Error", "Could not load the image!")
            return

        # Resize and preprocess image
        image = cv2.resize(image, (660, 380))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Extract ORB features
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        print(f"Keypoints detected: {len(keypoints) if keypoints else 0}")
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

        # Detect banknote
        result_text = self.detect_banknote(descriptors, keypoints)
        
        # Extract number with OCR
        number_text = extract_number_from_image(image)
        
        if "No Banknote Detected" not in result_text and "Error" not in result_text:
            cv2.rectangle(image_with_keypoints, (0, 0), (660, 380), (0, 255, 0), 3)

        # Convert image for Tkinter
        image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update GUI
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk
        self.result_label.config(text=result_text)
        self.number_label.config(text=f"Extracted Number: {number_text}")

    def detect_banknote(self, descriptors, keypoints):
        if descriptors is not None and len(keypoints) >= MIN_KEYPOINTS:
            descriptor_features = descriptors.flatten()[:27232]  # ফিচার সাইজ ফিক্সড
            expected_features = model.n_features_in_
            print(f"Descriptor length: {len(descriptor_features)}")
            print(f"Expected features: {expected_features}")

            if len(descriptor_features) < expected_features:
                descriptor_features = np.pad(descriptor_features, (0, expected_features - len(descriptor_features)), mode='constant')
            else:
                descriptor_features = descriptor_features[:expected_features]

            try:
                descriptor_features = descriptor_features.reshape(1, -1)
                probs = model.predict_proba(descriptor_features)[0]
                max_prob = np.max(probs)
                prediction = model.predict(descriptor_features)[0]
                print(f"Prediction index: {prediction}, Confidence: {max_prob}")
                print(f"Probabilities: {probs}")

                if max_prob >= CONFIDENCE_THRESHOLD:
                    predicted_value = labels_dict.get(int(prediction), "Unknown")
                    logging.info(f"Detected: {predicted_value} (Confidence: {max_prob:.2f})")
                    return f"Detected: {predicted_value} (Confidence: {max_prob:.2f})"
                else:
                    return "No Banknote Detected"
            except Exception as e:
                logging.error(f"Error in prediction: {e}")
                return "Prediction Error"
        else:
            return "No Banknote Detected"

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = BanknoteDetectorApp(root)
    root.mainloop()